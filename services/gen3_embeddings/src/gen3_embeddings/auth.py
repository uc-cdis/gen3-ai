"""
This service's authz resource convention.

The generic authorization machinery -- token handling, policy-engine calls, authz-mapping
retrieval, per-request caching, the HTTP-verb-to-CRUD mapping -- lives in `common.auth`.
What is specific to this service, and therefore lives here, is only:

- `AUTHZ`: the `AuthzConfig` binding this service's Arborist service name, token audience,
  and ArboristClient to those generic functions.
- the COLLECTION resource-path convention, in both directions: name to path, and a caller's
  granted paths back to collection names.

TWO KINDS OF AUTHZ RESOURCE
---------------------------

The convention in this module applies to COLLECTIONS ONLY. Do not read it as a constraint on
authz generally, because embeddings work differently and the difference matters:

- A COLLECTION's authz resource is DERIVED and FIXED: it is always
  `/vectorstore/collections/{collection_name}`, because the table has no authz column and a
  collection's name is its authz identity. That is why `collections`' RLS policy keys on
  `collection_name`, and why this module can translate between the two forms at all.

- An EMBEDDING's authz is STORED, ARBITRARY, and MUTABLE. It is whatever string the caller
  put in the `authz` column: `/programs/foo/projects/bar`, `/open`, anything. It defaults to
  the containing collection's path when a request does not supply one, but that is only a
  default, not a rule -- POST and PUT accept any `authz` in the body, and PUT can change an
  existing embedding's `authz` to a different value later. All the service requires is that
  the caller holds the relevant action on whatever path they name.

So `get_allowed_collection_names_from_authz` below deliberately narrows a caller's grants to
the collection-shaped ones, and its result feeds ONLY the `collections` policy. The
embeddings policy is fed the caller's grants UNNARROWED (see
`common.auth.get_allowed_authz_from_mapping`, which filters by service and method but never
by path shape), which is what lets an embedding carry an authz path this service knows
nothing about.

Routes do not call this module directly. They declare an action with
`gen3_embeddings.dependencies.authz`, which is the single place those two halves are
assembled. See the module docstring there for the layering.

NOTE on `ALLOW_ANONYMOUS_ACCESS`: `common.auth.authorize_request` returns early when it is
set, but `common.auth.get_user_authz_mapping` still requires a token. So enabling it does
not make this service readable anonymously -- resolving which collections a caller may see
goes through the mapping, which 401s without a token. `DEBUG_SKIP_AUTH` is the flag that
actually bypasses this service end to end.
"""

from collections.abc import Mapping

from starlette.requests import Request

from common.auth import AuthzConfig, get_allowed_authz_from_mapping
from gen3_embeddings import config
from gen3_embeddings.config import logging
from gen3_embeddings.models.helpers import normalize_collection_name

# Base path for COLLECTION resources. Not a prefix embedding authz has to sit under: see the
# module docstring.
AUTHZ_RESOURCE_BASE = "/vectorstore/collections"

# Audience this service expects in an access token. Pinned rather than derived from the
# request host, which is what `common.auth` does by default.
AUTHZ_TOKEN_AUDIENCE = "gen3"


def _arborist_client_from_request(request: Request | None):
    """
    Return the ArboristClient built at startup and held on the app state.

    Using the app-state client rather than a module-level one is what makes the configured
    `ARBORIST_URL` take effect, and lets tests substitute a client.

    Args:
        request (Request | None): The incoming request.

    Returns:
        ArboristClient: The client from `app.state`.

    Raises:
        Exception: If there is no request to read the app state from.
    """
    if not request:
        raise Exception("Expected a request, got None. Cannot determine Arborist Client from app state from request.")

    return request.app.state.arborist_client


AUTHZ = AuthzConfig(
    service_name=config.AUTHZ_SERVICE_NAME,
    audience=AUTHZ_TOKEN_AUDIENCE,
    arborist_resolver=_arborist_client_from_request,
)


def get_authz_resource_path_from_collection_name(collection_name: str) -> str:
    """
    Build the Arborist resource path for a vector collection.

    Args:
        collection_name (str): Collection name, or "" for the base resource.

    Returns:
        str: The resource path.
    """
    if collection_name == "":
        return AUTHZ_RESOURCE_BASE
    return f"{AUTHZ_RESOURCE_BASE}/{collection_name}"


def get_allowed_collection_names_from_authz(allowed_authz: list[str]) -> set[str]:
    """
    Derive the collection names a caller may act on from their allowed authz paths.

    Follows the resource convention:

      /vectorstore/collections
      /vectorstore/collections/{collection_name}

    Only names that are ALREADY in canonical form are kept. Nothing is rewritten here:
    Arborist resource paths are case-sensitive, so normalizing a grant would mean this
    service disagreed with the policy engine about what the grant says. But collection names
    are always stored normalized, so a resource naming `Docs`, `my collection` or
    `bad.name` cannot match any stored collection and therefore grants nothing. Those are
    dropped rather than carried into every comparison.

    The practical consequence is that a policy must name the collection exactly as it is
    stored, in lowercase.

    Args:
        allowed_authz (list[str]): Authz resource paths the caller holds.

    Returns:
        set[str]: Canonical collection names the caller may act on. Empty is a valid
        fail-closed result meaning "no collections", not "all collections".
    """
    allowed: set[str] = set()

    for item in allowed_authz:
        if not isinstance(item, str):
            continue
        if item == AUTHZ_RESOURCE_BASE:
            # base resource: may mean "can access all collections", depending on policy
            # for now, we'll pass
            continue
        if item.startswith(AUTHZ_RESOURCE_BASE + "/"):
            # e.g. "/vectorstore/collections/my_collection"
            parts = item.split("/")
            if parts:
                if len(parts) != 4:
                    # # Expect exactly: ["", "vectorstore", "collections", "{collection_name}"]
                    # This covers "/vectorstore/collections/a/b" (len=5), etc.
                    continue
                name = parts[-1]
                if not name:
                    continue
                try:
                    # keep only names already in canonical form. Comparing rather than
                    # replacing catches uppercase and padding as well as invalid characters,
                    # without rewriting what the policy engine said.
                    is_canonical = normalize_collection_name(name) == name
                except ValueError:
                    is_canonical = False
                if not is_canonical:
                    # cannot match a stored collection, so it grants nothing. Skip rather
                    # than raise, so one odd policy entry cannot fail every request.
                    logging.debug(f"Ignoring authz resource that cannot name a stored collection: {item}")
                    continue
                allowed.add(name)
    return allowed


def get_allowed_collection_names_from_mapping(authz_mapping: Mapping, method: str) -> set[str]:
    """
    Go straight from a policy-engine mapping to allowed collection names for one action.

    Composes `common.auth.get_allowed_authz_from_mapping` with this service's path
    convention. Provided for callers that already hold a mapping; request-scoped code should
    use `gen3_embeddings.dependencies.authz` instead, which caches the mapping.

    Args:
        authz_mapping (Mapping): Mapping as returned by the policy engine.
        method (str): Logical CRUD action, e.g. "read".

    Returns:
        set[str]: Canonical collection names the caller may act on with `method`.
    """
    return get_allowed_collection_names_from_authz(
        get_allowed_authz_from_mapping(authz_mapping, method=method, service=AUTHZ.service_name)
    )
