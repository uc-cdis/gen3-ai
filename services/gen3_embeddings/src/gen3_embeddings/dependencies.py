"""
FastAPI dependencies that bridge the authorization layer and the data access layer.

These live outside `database/db.py` on purpose. Resolving a caller's authz means talking to
the Gen3 policy engine over the network, and the data access layer should only ever talk to
the database. Keeping the wiring here means `db.py` does not import `auth` at all: routes
declare which authz context they need, and the DAL just receives values.
"""

from fastapi import Depends, Request

from gen3_embeddings.auth import (
    get_allowed_authz_for_request,
    get_allowed_authz_for_request_with_method,
    get_allowed_collection_names_from_authz,
)
from gen3_embeddings.database.db import DataAccessLayer, get_pool


async def get_data_access_layer(request: Request):
    """
    Yield a DAL scoped to the authz the caller holds for this request's HTTP method.

    The method is derived from the verb (GET -> read, POST -> create, PUT/PATCH -> update,
    DELETE -> delete), so the DAL only ever sees rows the caller holds that specific
    permission on.

    Args:
        request (Request): Incoming request, used to resolve the caller's authz mapping.

    Yields:
        DataAccessLayer: DAL bound to the connection pool and the caller's allowed authz.
    """
    pool = await get_pool()
    allowed_authz = await get_allowed_authz_for_request(request)
    dal = DataAccessLayer(pool, allowed_authz=allowed_authz)
    yield dal


async def get_data_access_layer_for_read_operations(request: Request):
    """
    Yield a DAL scoped to the caller's `read` authz, regardless of HTTP method.

    Use this for endpoints that read data but are declared POST because the query does not
    fit in a query string (bulk reads and search). Depending on `get_data_access_layer`
    there would scope the DAL to `create` permissions instead, authorizing the wrong action.

    Args:
        request (Request): Incoming request, used to resolve the caller's authz mapping.

    Yields:
        DataAccessLayer: DAL bound to the connection pool and the caller's `read` authz.
    """
    pool = await get_pool()
    allowed_authz = await get_allowed_authz_for_request_with_method(request, method="read")
    dal = DataAccessLayer(pool, allowed_authz=allowed_authz)
    yield dal


async def get_allowed_collection_names(request: Request) -> set[str]:
    """
    Resolve the collection names the caller may act on with this request's HTTP method.

    The `collections` table has no RLS policy, so unlike embeddings it cannot be filtered by
    the database. Routes pass this set into the DAL, which filters by it. That keeps the
    authorization decision at the HTTP boundary while the DAL stays a pure data layer.

    Args:
        request (Request): Incoming request, used to resolve the caller's authz mapping.

    Returns:
        set[str]: Allowed collection names. Empty means none, which is fail-closed.
    """
    allowed_authz = await get_allowed_authz_for_request(request)
    return get_allowed_collection_names_from_authz(allowed_authz)


async def get_allowed_collection_names_for_read_operations(request: Request) -> set[str]:
    """
    Resolve the collection names the caller may READ, regardless of HTTP method.

    The read-scoped counterpart to `get_allowed_collection_names`, for the POST endpoints
    that only read (bulk reads and search).

    Args:
        request (Request): Incoming request, used to resolve the caller's authz mapping.

    Returns:
        set[str]: Allowed collection names for reading. Empty means none.
    """
    allowed_authz = await get_allowed_authz_for_request_with_method(request, method="read")
    return get_allowed_collection_names_from_authz(allowed_authz)


# Route-facing aliases, so endpoint signatures read as declarations of what they need.
AllowedCollectionNames = Depends(get_allowed_collection_names)
AllowedCollectionNamesForRead = Depends(get_allowed_collection_names_for_read_operations)
