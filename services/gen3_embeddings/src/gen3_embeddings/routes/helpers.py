"""Helpers for registering route path operations."""

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter

# Keyword arguments that only affect the generated OpenAPI document, so they are not copied
# onto the undocumented trailing-slash alias.
_DOC_ONLY_KWARGS = frozenset(
    {
        "summary",
        "description",
        "responses",
        "tags",
        "include_in_schema",
        "operation_id",
        "deprecated",
    }
)


def dual_path(router: APIRouter, method: str, path: str, **kwargs: Any) -> Callable:
    """
    Register one path operation at both `path` and `path/`, documenting only `path`.

    Callers reach these endpoints with and without a trailing slash, and Starlette's
    automatic redirect is not usable here because clients would have to follow a 307 and
    re-send their Authorization header. So both spellings are registered as real routes.

    Registering them as two separate decorators is what this replaces: the alias only ever
    received `include_in_schema=False`, silently dropping `response_model`,
    `response_model_exclude_none` and `status_code`. That made `/path/` return fields the
    documented response model excludes, and made DELETE aliases answer 200 instead of 204.
    Deriving the alias from the same kwargs means the two cannot diverge.

    Args:
        router (APIRouter): Router to register on.
        method (str): HTTP method name as it appears on the router, e.g. "get" or "post".
        path (str): Canonical path, WITHOUT a trailing slash.
        **kwargs: Passed to the router. Everything that affects behavior is applied to both
            routes; OpenAPI-only arguments are applied to the documented route alone.

    Returns:
        Callable: Decorator that registers the endpoint function on both paths.
    """
    alias_kwargs = {key: value for key, value in kwargs.items() if key not in _DOC_ONLY_KWARGS}
    alias_kwargs["include_in_schema"] = False

    def decorator(func: Callable) -> Callable:
        # register the alias first, matching the original stacked-decorator ordering
        getattr(router, method)(f"{path}/", **alias_kwargs)(func)
        return getattr(router, method)(path, **kwargs)(func)

    return decorator
