"""
Enforcement of the request limits configured in `gen3_embeddings.config`.

Two layers, because they can only be applied at different points:

- `RequestSizeLimitMiddleware` bounds the request in BYTES, before the application runs.
  This has to be ASGI middleware rather than a dependency or a validator: the field limits
  in `models.schemas` are checked against parsed Python objects, and the parse is itself
  the expensive step. A 1 GiB JSON array costs a 1 GiB read plus the memory of the
  resulting list before any validator is reached, so by then the damage is done.

- The helpers here bound the shape of values Pydantic cannot express as a constraint,
  namely metadata size and nesting depth.

Everything else is a declarative `Field` constraint on the schema or path operation.
"""

import json
from collections.abc import Awaitable, Callable
from typing import Any

from starlette.datastructures import Headers

from gen3_embeddings.config import (
    MAX_METADATA_BYTES,
    MAX_METADATA_DEPTH,
    MAX_METADATA_KEYS,
    logging,
)

Scope = dict[str, Any]
Message = dict[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]


class RequestSizeLimitMiddleware:
    """
    Reject request bodies larger than `max_body_bytes` before they are read into memory.

    Implemented as raw ASGI middleware rather than a `BaseHTTPMiddleware` subclass so that
    it sits ahead of the exception middleware and can refuse a request without the body
    ever being buffered by the application.
    """

    def __init__(self, app: Callable, max_body_bytes: int) -> None:
        """
        Args:
            app (Callable): The ASGI application to wrap.
            max_body_bytes (int): Largest request body to accept, in bytes.
        """
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """
        Pass the request through, or answer 413 if its body is too large.

        Args:
            scope (Scope): ASGI connection scope.
            receive (Receive): ASGI receive callable.
            send (Send): ASGI send callable.
        """
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        content_length = Headers(scope=scope).get("content-length")

        if content_length is not None:
            try:
                declared_bytes = int(content_length)
            except ValueError:
                await self._respond(send, 400, "Invalid Content-Length header.")
                return

            if declared_bytes > self.max_body_bytes:
                await self._reject(scope, send, declared_bytes)
                return

            # The ASGI server holds the client to the length it declared, so a body that
            # fits the header cannot grow past it while streaming. Nothing left to police.
            await self.app(scope, receive, send)
            return

        # No Content-Length, so the size is unannounced and we have to count as we read.
        # Requests without a body still get here (their first message is simply empty),
        # which costs one no-op receive.
        body, exceeded_by = await self._read_capped(receive)
        if exceeded_by is not None:
            await self._reject(scope, send, exceeded_by)
            return

        await self.app(scope, _replay_body(body), send)

    async def _read_capped(self, receive: Receive) -> tuple[bytes, int | None]:
        """
        Read the body, stopping as soon as it exceeds the limit.

        Args:
            receive (Receive): ASGI receive callable.

        Returns:
            tuple[bytes, int | None]: The body, and None if it fit. If the limit was
            exceeded, the size read so far is returned instead and the body is discarded,
            so at most `max_body_bytes` plus one chunk is ever held.
        """
        chunks: list[bytes] = []
        total = 0

        while True:
            message = await receive()

            if message["type"] == "http.disconnect":
                return b"", None

            chunk = message.get("body", b"")
            total += len(chunk)
            if total > self.max_body_bytes:
                return b"", total

            chunks.append(chunk)

            if not message.get("more_body", False):
                return b"".join(chunks), None

    async def _reject(self, scope: Scope, send: Send, size: int) -> None:
        """
        Log and answer 413 for an over-large body.

        Args:
            scope (Scope): ASGI connection scope, used for the log line.
            send (Send): ASGI send callable.
            size (int): Size that tripped the limit, in bytes.
        """
        logging.warning(
            "Rejecting %s %s: request body of %d bytes exceeds the %d byte limit.",
            scope.get("method", "?"),
            scope.get("path", "?"),
            size,
            self.max_body_bytes,
        )
        await self._respond(
            send,
            413,
            f"Request body is too large. The limit is {self.max_body_bytes} bytes. "
            "Split the request into smaller batches.",
        )

    @staticmethod
    async def _respond(send: Send, status_code: int, detail: str) -> None:
        """
        Send a JSON error response, matching the shape FastAPI uses for `HTTPException`.

        Args:
            send (Send): ASGI send callable.
            status_code (int): HTTP status code to return.
            detail (str): Message for the `detail` field.
        """
        body = json.dumps({"detail": detail}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status_code,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    # The client has nothing to gain from reusing a connection whose
                    # request we stopped reading partway through.
                    (b"connection", b"close"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


def _replay_body(body: bytes) -> Receive:
    """
    Build a receive callable that serves an already-read body to the application.

    Args:
        body (bytes): The buffered request body.

    Returns:
        Receive: A callable yielding the body once, then disconnects.
    """
    sent = False

    async def receive() -> Message:
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    return receive


def metadata_depth(value: Any) -> int:
    """
    Return the nesting depth of a decoded JSON value.

    Walks breadth-first and stops one level past the limit, so a pathologically nested
    document costs no more to measure than a compliant one. Iterative rather than
    recursive for the same reason.

    Args:
        value (Any): Decoded JSON value to measure.

    Returns:
        int: Nesting depth, where a scalar is 0 and `{"a": 1}` is 1. Capped at
        `MAX_METADATA_DEPTH + 1`, which is enough to tell that the limit was exceeded.
    """
    depth = 0
    frontier = [value]

    while frontier:
        children: list[Any] = []
        for item in frontier:
            if isinstance(item, dict):
                children.extend(item.values())
            elif isinstance(item, list):
                children.extend(item)

        if not children:
            # Nothing at this level nests any further.
            break

        depth += 1
        if depth > MAX_METADATA_DEPTH:
            return depth
        frontier = children

    return depth


def validate_metadata(metadata: dict | None) -> dict | None:
    """
    Check caller-supplied embedding metadata against the configured limits.

    Metadata is stored as jsonb, hashed on write, and returned in full on every read of its
    embedding, so its serialized size is the bound that matters; the key and depth limits
    exist to give a clearer error than a byte count alone would.

    Args:
        metadata (dict | None): Metadata as supplied by the caller.

    Returns:
        dict | None: The metadata unchanged, so this can be used as a Pydantic validator.

    Raises:
        ValueError: If the metadata has too many top-level keys, nests too deeply, or is
            too large once serialized.
    """
    if metadata is None:
        return None

    if len(metadata) > MAX_METADATA_KEYS:
        raise ValueError(f"metadata may have at most {MAX_METADATA_KEYS} top-level keys, got {len(metadata)}")

    depth = metadata_depth(metadata)
    if depth > MAX_METADATA_DEPTH:
        raise ValueError(f"metadata may nest at most {MAX_METADATA_DEPTH} levels deep")

    # Measured on the serialization rather than the object graph because that is what gets
    # stored, hashed, and sent back.
    size = len(json.dumps(metadata, default=str))
    if size > MAX_METADATA_BYTES:
        raise ValueError(f"metadata may be at most {MAX_METADATA_BYTES} bytes when serialized, got {size}")

    return metadata
