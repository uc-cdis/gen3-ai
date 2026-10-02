"""
Translation from data access layer errors into HTTP responses.

This is the only place that knows how a `DataAccessError` should surface over HTTP. Keeping
the mapping here rather than on the exceptions themselves means the DAL carries no HTTP
concerns, while every route still gets consistent status codes without its own try/except.

It also replaces FastAPI's request validation handler, only so that a 422 can always be
serialized; see `_json_safe`.
"""

import json
import math
from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette import status

from gen3_embeddings.config import logging
from gen3_embeddings.database.errors import (
    CollectionAlreadyExistsError,
    CollectionCreateFailedError,
    CollectionNameNotAllowedError,
    DataAccessError,
    DuplicateEmbeddingError,
    EmbeddingDimensionMismatchError,
    EmbeddingNotRepresentableError,
    EmbeddingsAlreadyExistError,
    EmbeddingWriteInconsistencyError,
    InvalidCollectionNameError,
    MetadataLengthMismatchError,
    RowLevelSecurityDeniedError,
)

# Anything not listed here falls back to 500, so a new DataAccessError cannot accidentally
# be reported as a client error.
DATA_ACCESS_ERROR_STATUS: dict[type[DataAccessError], int] = {
    InvalidCollectionNameError: status.HTTP_400_BAD_REQUEST,
    CollectionCreateFailedError: status.HTTP_400_BAD_REQUEST,
    MetadataLengthMismatchError: status.HTTP_400_BAD_REQUEST,
    EmbeddingDimensionMismatchError: status.HTTP_400_BAD_REQUEST,
    EmbeddingNotRepresentableError: status.HTTP_400_BAD_REQUEST,
    CollectionNameNotAllowedError: status.HTTP_403_FORBIDDEN,
    RowLevelSecurityDeniedError: status.HTTP_403_FORBIDDEN,
    CollectionAlreadyExistsError: status.HTTP_409_CONFLICT,
    EmbeddingsAlreadyExistError: status.HTTP_409_CONFLICT,
    DuplicateEmbeddingError: status.HTTP_409_CONFLICT,
    EmbeddingWriteInconsistencyError: status.HTTP_500_INTERNAL_SERVER_ERROR,
}


def get_status_code_for_error(exc: DataAccessError) -> int:
    """
    Resolve the HTTP status code for a data access error.

    Walks the exception's class hierarchy so a subclass inherits its parent's mapping.

    Args:
        exc (DataAccessError): The raised error.

    Returns:
        int: The mapped status code, or 500 if the type is not mapped.
    """
    for _class in type(exc).__mro__:
        # the issubclass guard is also what narrows `klass` from `type` for the type checker,
        # since only DataAccessError subclasses can be keys
        if issubclass(_class, DataAccessError) and _class in DATA_ACCESS_ERROR_STATUS:
            return DATA_ACCESS_ERROR_STATUS[_class]
    return status.HTTP_500_INTERNAL_SERVER_ERROR


def _json_safe(value: Any) -> Any:
    """
    Replace non-finite floats with the JSON token the caller sent, so the value can be encoded.

    A 422 echoes the rejected `input` back. The stdlib parser Starlette uses accepts `NaN`
    and `Infinity`, so a request refused for holding one carries it into that echo, and
    `JSONResponse` refuses to encode it -- turning the client's 422 into a 500. Rendering it
    as the string `"NaN"` keeps the response shape and still shows the caller what was wrong.

    Args:
        value (Any): A `jsonable_encoder` result: dicts, lists, and scalars.

    Returns:
        Any: `value` with every non-finite float replaced by its JSON token as a string.
    """
    if isinstance(value, float) and not math.isfinite(value):
        # json.dumps renders these as NaN / Infinity / -Infinity, the tokens the caller sent
        return json.dumps(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def register_error_handlers(app: FastAPI) -> None:
    """
    Register the data access and request validation error handlers on the app.

    Args:
        app (FastAPI): The application to register on.
    """

    @app.exception_handler(DataAccessError)
    async def data_access_error_handler(request: Request, exc: DataAccessError) -> JSONResponse:
        status_code = get_status_code_for_error(exc)

        # Server-side faults are our bug, not the caller's, so make them loud. Client errors
        # are expected traffic and stay at debug.
        if status_code >= status.HTTP_500_INTERNAL_SERVER_ERROR:
            logging.error(f"{type(exc).__name__} on {request.method} {request.url.path}: {exc}")
        else:
            logging.debug(f"{type(exc).__name__} on {request.method} {request.url.path}: {exc}")

        return JSONResponse(status_code=status_code, content={"detail": str(exc)})

    @app.exception_handler(RequestValidationError)
    async def request_validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        # FastAPI's default handler, plus `_json_safe`
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={"detail": _json_safe(jsonable_encoder(exc.errors()))},
        )
