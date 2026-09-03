"""
The single place where authorization is assembled for a request.

LAYERING
--------

Every endpoint's authorization is one declaration in its signature:

    ctx: AuthzContext = Depends(authz("read"))

That declaration is the whole authorization decision for the route, and this module is the
only thing that acts on it. The split is:

- The ROUTE decides which action it performs and, through its path, which resource. It does
  web work: parsing, status codes, response shaping.
- This module turns (action, resource) into (a policy-engine check, an authz-scoped DAL).
- The DAL runs SQL and sets the row-level security context. It resolves nothing.
- POSTGRES enforces row visibility, via RLS policies on both `embeddings_*` and
  `collections`.

WHY THE ACTION IS DECLARED, NOT INFERRED
----------------------------------------

It used to be derived from the HTTP verb. That is right for most routes and silently wrong
for the ones that read but are declared POST because their query does not fit in a query
string (search, bulk reads): the verb says "create", so the caller was authorized for the
wrong action. The previous fix was a second set of dependencies
(`get_data_access_layer_for_read_operations` and friends) that a route had to remember to
pick, in matching pairs, with nothing checking that it did.

Declaring the action removes both problems: there is one dependency per route, the verb is
not consulted, and a route that reads says `authz("read")` whatever its verb is.

WHAT THE DEPENDENCY DOES
------------------------

1. Normalizes the `collection_name` path parameter, if the route has one, so the resource
   path checked and the name compared in the database are the same string.
2. Resolves the caller's authz resources for the declared action, from the policy engine.
3. If the route names a collection, asks the policy engine whether the caller may perform
   the declared action (and any `also_require` actions) on that collection's resource path.
   Routes with no collection in their path -- cross-collection search, bulk read, listing
   and creating collections -- have no single resource to check, so they are scoped by step
   2 and by RLS instead. That is why an unauthorized cross-collection search returns an
   empty result rather than a 403: nothing was denied, there was simply nothing to see.
4. Hands the resolved authz to a `DataAccessLayer`, which is the only consumer of it.

Handlers that need an extra check the path cannot express -- a caller-supplied `authz` in a
request body -- call `ctx.require(...)`, which is the same policy-engine check against a
resource resolved at runtime.

A NOTE ON EMBEDDING AUTHZ
-------------------------

Step 3 checks a COLLECTION resource, which is always
`/vectorstore/collections/{collection_name}`. An embedding's own `authz` is not: it is an
arbitrary, mutable string in the row, defaulting to its collection's path but free to be
anything the caller holds the action on, and changeable afterwards by PUT. That is exactly
what `ctx.require(...)` exists for -- the resource is not knowable from the path, so the
handler resolves it from the body and asks separately.

`allowed_authz` here is therefore NOT narrowed to collection-shaped paths. It is every
resource the caller holds the declared action on, which is what the embeddings RLS policy
compares each row's `authz` against. `allowed_collection_names` is the narrowed form, and it
feeds only the `collections` policy.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import asyncpg
from fastapi import Request

from common.auth import authorize_request, get_allowed_authz_for_request
from gen3_embeddings.auth import (
    AUTHZ,
    get_allowed_collection_names_from_authz,
    get_authz_resource_path_from_collection_name,
)
from gen3_embeddings.database.db import DataAccessLayer
from gen3_embeddings.params import normalized_collection_name

AuthzAction = Literal["read", "create", "update", "delete"]
"""The logical actions this service authorizes. These are the Arborist access methods."""


def db_pool_from_request(request: Request) -> asyncpg.Pool:
    """
    Return the connection pool built at startup and held on the app state.

    Deliberately does not fall back to creating one. A lazily created pool would be built
    per-event-loop-race rather than once, and would silently paper over an app whose lifespan
    never ran, which is also an app that never verified row-level security is in effect.

    Args:
        request (Request): The incoming request.

    Returns:
        asyncpg.Pool: The pool from `app.state`.

    Raises:
        Exception: If the app has no pool, meaning the lifespan handler did not run.
    """
    pool = getattr(request.app.state, "db_pool", None)
    if pool is None:
        raise Exception("No database pool on app state. The application lifespan did not run.")

    return pool


@dataclass(frozen=True)
class AuthzContext:
    """
    Everything a handler needs to act on the caller's behalf, for one declared action.

    Attributes:
        action (AuthzAction): The action the route declared. The DAL's row-level security
            scope is this action's resource set, so a route declaring "read" cannot write
            even if its handler tried to.
        allowed_authz (list[str]): Every authz resource path the caller holds `action` on,
            whatever its shape -- what the embeddings policy matches each row's `authz`
            against. Empty is a valid fail-closed result.
        allowed_collection_names (set[str]): Those same grants narrowed to the
            collection-shaped ones and reduced to names, for the `collections` policy.
        dal (DataAccessLayer): Data access layer bound to the two fields above.
        collection_name (str | None): Normalized collection name from the route's path, or
            None if the route does not name one.
        request (Request): The incoming request, for follow-up checks.
    """

    action: AuthzAction
    allowed_authz: list[str]
    allowed_collection_names: set[str]
    dal: DataAccessLayer
    collection_name: str | None
    request: Request

    async def require(self, *resources: str, action: AuthzAction | None = None) -> None:
        """
        Check the caller may perform `action` on resources not derivable from the path.

        This is for authz paths that arrive in a request body, where the resource is only
        known once the body is parsed. It is the same policy-engine check the dependency
        already performed on the path's resource.

        Args:
            *resources: Authz resource paths to check. Passing none is a no-op.
            action (AuthzAction | None): Action to check, defaulting to the declared action.

        Raises:
            HTTPException: 403 if the caller lacks the access, 401 if the token is invalid.
        """
        if not resources:
            return

        await authorize_request(
            authz_resources=list(resources),
            authz_access_method=action or self.action,
            request=self.request,
            authz_config=AUTHZ,
        )


@dataclass(frozen=True)
class AuthzDependency:
    """
    The callable a route declares, carrying the declaration it was built from.

    A callable object rather than a closure so that `action` and `also_require` are declared
    fields of a real type. Tests read them off a built app to assert that every route has
    exactly one authz declaration and that it is the right one -- an unguarded route is
    otherwise invisible until someone calls it -- and attributes stapled onto a function
    would be invisible to a type checker on both sides.

    Attributes:
        action (AuthzAction): The action the route performs.
        also_require (tuple[AuthzAction, ...]): Further actions demanded on the route's
            collection resource.
    """

    action: AuthzAction
    also_require: tuple[AuthzAction, ...] = ()

    async def __call__(self, request: Request) -> AuthzContext:
        """
        Resolve this request's authorization.

        Args:
            request (Request): The incoming request. The only dependency, so this works
                unchanged whatever else a route's signature declares.

        Returns:
            AuthzContext: The caller's resolved authz and a DAL scoped to it.

        Raises:
            HTTPException: 400 if the path's collection name is invalid, 401 if the token is
                missing or invalid, 403 if the caller lacks the declared action on the
                route's collection.
        """
        # Normalized here, not read raw, so the resource path checked against the policy
        # engine is byte-identical to the name compared in the database. Arborist paths are
        # case-sensitive, so an unnormalized path could be denied by one gate and accepted
        # by the other.
        raw_collection_name = request.path_params.get("collection_name")
        collection_name = normalized_collection_name(raw_collection_name) if raw_collection_name is not None else None

        allowed_authz = await get_allowed_authz_for_request(request, method=self.action, authz_config=AUTHZ)

        # Deny before taking a connection out of the pool.
        if collection_name is not None:
            resource = get_authz_resource_path_from_collection_name(collection_name)
            for required_action in (self.action, *self.also_require):
                await authorize_request(
                    authz_resources=[resource],
                    authz_access_method=required_action,
                    request=request,
                    authz_config=AUTHZ,
                )

        allowed_collection_names = get_allowed_collection_names_from_authz(allowed_authz)

        dal = DataAccessLayer(
            db_pool_from_request(request),
            allowed_authz=allowed_authz,
            allowed_collection_names=allowed_collection_names,
        )

        return AuthzContext(
            action=self.action,
            allowed_authz=allowed_authz,
            allowed_collection_names=allowed_collection_names,
            dal=dal,
            collection_name=collection_name,
            request=request,
        )


def authz(
    action: AuthzAction,
    *,
    also_require: Sequence[AuthzAction] = (),
) -> AuthzDependency:
    """
    Build the dependency that authorizes a route for `action`.

    Args:
        action (AuthzAction): The action this route performs. Drives both the policy-engine
            check and the row-level security scope the DAL runs under.
        also_require (Sequence[AuthzAction]): Additional actions the caller must hold on the
            route's collection resource. For an upsert, which both creates and updates, the
            primary action is what scopes RLS and this is the rest of what is demanded.

    Returns:
        AuthzDependency: The FastAPI dependency yielding an `AuthzContext`.
    """
    return AuthzDependency(action=action, also_require=tuple(also_require))
