"""
Reusable OpenAPI `responses=` entries for FastAPI path operation decorators.

These keep the error documentation in the generated public API docs consistent
across services, and spare every route from repeating the same dicts.

Only document responses a caller should actually handle. In particular there is
deliberately no 500 entry here: an internal server error is a bug, not part of
the contract, and documenting it invites clients to build against it.

Usage:
    @router.get(
        "/things/{thing_name}",
        summary="Read thing",
        description="Returns the thing.",
        responses={**AUTH_RESPONSES, **not_found_response("Thing")},
    )
"""

from types import MappingProxyType
from typing import Any

from starlette import status

# Mappings are exposed read-only so a caller spreading them with `**` cannot
# accidentally mutate the shared copy for every other route in the process.

AUTH_RESPONSES: MappingProxyType[int, dict[str, Any]] = MappingProxyType(
    {
        status.HTTP_401_UNAUTHORIZED: {"description": "User unauthenticated"},
        status.HTTP_403_FORBIDDEN: {"description": "User does not have access"},
    }
)
"""The 401/403 pair every authenticated route can return, documented as one block."""

BAD_REQUEST_RESPONSE: MappingProxyType[int, dict[str, Any]] = MappingProxyType(
    {
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid request payload or parameters"},
    }
)
"""For routes that reject malformed input beyond what schema validation catches."""

NO_CONTENT_RESPONSE: MappingProxyType[int, dict[str, Any]] = MappingProxyType(
    {
        status.HTTP_204_NO_CONTENT: {"description": "Successful operation, no content returned"},
    }
)
"""For deletes and other operations that succeed without a response body."""


def not_found_response(resource: str) -> dict[int, dict[str, Any]]:
    """
    Build a 404 entry naming the specific resource that could not be found.

    A generic "Resource not found" is less useful in rendered docs than knowing
    whether it is the collection or the embedding that has to exist.

    Args:
        resource (str): Human-readable resource name, e.g. "Collection".

    Returns:
        dict[int, dict[str, Any]]: A `responses=` fragment to spread into a decorator.
    """
    return {status.HTTP_404_NOT_FOUND: {"description": f"{resource} not found"}}
