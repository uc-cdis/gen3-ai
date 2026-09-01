"""
Common authentication and Gen3 policy-engine (Arborist) authorization.

WHAT LIVES HERE
---------------

Everything about authorization that is not specific to one service's resource layout:

- reading and validating the caller's token (`get_user_id`, `get_username`)
- asking the policy engine a yes/no question about named resources (`authorize_request`)
- fetching the caller's full authz mapping and reducing it to "the resources you hold
  `method` on" (`get_user_authz_mapping`, `get_allowed_authz_from_mapping`,
  `get_allowed_authz_for_request`)
- the one canonical HTTP-verb-to-CRUD-action mapping (`crud_action_for_method`)
- per-request caching of all of the above, so a request that needs several of these costs
  at most one policy-engine round trip per distinct question

WHAT DOES NOT LIVE HERE
-----------------------

How a service names its resources. A service defines its own resource-path convention and
its own dependencies, and passes an `AuthzConfig` in. See
`gen3_embeddings.auth`/`gen3_embeddings.dependencies` for a worked example: the service
supplies the `/vectorstore/collections/{name}` convention and this module answers questions
about it.

CONFIGURATION
-------------

`AuthzConfig` carries the three things that legitimately differ between services:

- `service_name`: the Arborist "service" the checks are scoped to.
- `audience`: expected token audience. `None` derives it from the request as
  `https://{host}/user`, which is what the Gen3 AuthN/Z service issues. A service that
  needs a fixed audience passes it explicitly.
- `arborist_resolver`: how to get an ArboristClient. `None` uses the module-level singleton.
  Services that build a client at startup pass a resolver reading it off `app.state`, so the
  client's configured base URL and connection pool are actually used.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from authutils.token.fastapi import access_token
from fastapi import HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from gen3authz.client.arborist.async_client import ArboristClient
from gen3authz.client.arborist.errors import ArboristError
from starlette.status import HTTP_401_UNAUTHORIZED as HTTP_401_UNAUTHENTICATED
from starlette.status import (
    HTTP_403_FORBIDDEN,
    HTTP_500_INTERNAL_SERVER_ERROR,
)

from common import config
from common.config import logging

get_bearer_token = HTTPBearer(auto_error=False)
arborist = ArboristClient()

# The logical CRUD action each HTTP verb corresponds to. This is the single definition of
# that mapping; a route that reads but is declared POST (because its query does not fit in a
# query string) must NOT rely on it, and should declare its action explicitly instead.
CRUD_ACTION_BY_HTTP_METHOD: dict[str, str] = {
    "GET": "read",
    "HEAD": "read",
    "POST": "create",
    "PUT": "update",
    "PATCH": "update",
    "DELETE": "delete",
}

# Prefix for everything this module caches on `request.state`, so it cannot collide with
# whatever a service puts there.
_CACHE_PREFIX = "_gen3_authz_"


def crud_action_for_method(http_method: str) -> str:
    """
    Return the logical CRUD action for an HTTP verb.

    Args:
        http_method (str): HTTP method name, any case.

    Returns:
        str: "read", "create", "update", "delete", or "unknown" for an unmapped verb.
            "unknown" is deliberately not an alias for anything permissive: it will match no
            policy, so an unmapped verb denies rather than allows.
    """
    return CRUD_ACTION_BY_HTTP_METHOD.get(http_method.upper(), "unknown")


ArboristResolver = Callable[[Request | None], ArboristClient]


@dataclass(frozen=True)
class AuthzConfig:
    """
    A service's authorization settings, passed to the functions in this module.

    Attributes:
        service_name (str): Arborist service name these checks are scoped to.
        audience (str | None): Expected token audience, or None to derive
            `https://{request host}/user` from the request.
        arborist_resolver (ArboristResolver | None): Returns the ArboristClient to use, or
            None to use this module's singleton.
    """

    service_name: str
    audience: str | None = None
    arborist_resolver: ArboristResolver | None = None

    def arborist_client(self, request: Request | None) -> ArboristClient:
        """
        Resolve the ArboristClient for this request.

        Args:
            request (Request | None): The incoming request, passed to the resolver.

        Returns:
            ArboristClient: The client to use for policy-engine calls.
        """
        if self.arborist_resolver is None:
            return arborist
        return self.arborist_resolver(request)

    def expected_audience(self, request: Request | None) -> str | None:
        """
        Resolve the token audience to validate against.

        Args:
            request (Request | None): Used to derive the audience when none is configured.

        Returns:
            str | None: The configured audience, the request-derived Gen3 audience, or None
            if neither is available.
        """
        if self.audience is not None:
            return self.audience
        if request:
            # This is what the Gen3 AuthN/Z service adds as the audience to represent Gen3
            # services
            return f"https://{request.base_url.netloc}/user"
        logging.warning(
            "Unable to determine expected audience b/c request context was not provided... setting audience to `None`."
        )
        return None


def _cache_get(request: Request | None, key: str) -> Any:
    """Read a per-request cached value, or None when there is no request to cache on."""
    if not request:
        return None
    return getattr(request.state, f"{_CACHE_PREFIX}{key}", None)


def _cache_set(request: Request | None, key: str, value: Any) -> None:
    """Store a per-request cached value, if there is a request to cache on."""
    if request:
        setattr(request.state, f"{_CACHE_PREFIX}{key}", value)


async def authorize_request(
    authz_resources: list[str],
    authz_service_name: str | None = None,
    authz_access_method: str = "access",
    token: HTTPAuthorizationCredentials | None = None,
    request: Request | None = None,
    authz_config: AuthzConfig | None = None,
):
    """
    Authorizes the incoming request based on the provided token and Arborist access policies.

    Args:
        authz_resources (list[str]): The list of resources to check against.
        authz_service_name (str | None): The Arborist service to supply for the check.
            Ignored when `authz_config` is given, which carries its own service name.
        authz_access_method (str): The Arborist access method to check (default is "access").
        token (HTTPAuthorizationCredentials): an authorization token (optional, you can also provide request
            and this can be parsed from there). This has priority over any token from request.
        request (Request): The incoming HTTP request. Used to parse tokens from header.
        authz_config (AuthzConfig | None): Service authz settings. Supplies the service name,
            the expected audience, and which ArboristClient to use.

    Raises:
        HTTPException: 401 if the token is missing or invalid, 403 if the caller lacks the
            access, 500 if the policy engine cannot be reached.

    Note:
        If `ALLOW_ANONYMOUS_ACCESS` is enabled, authorization check is bypassed. If `DEBUG_SKIP_AUTH` is enabled
        and no token is provided, the check is also bypassed.
    """
    if config.ALLOW_ANONYMOUS_ACCESS:
        logging.debug("ALLOW_ANONYMOUS_ACCESS mode is on, BYPASSING authorization check")
        return

    if config.DEBUG_SKIP_AUTH and not token:
        logging.warning("DEBUG_SKIP_AUTH mode is on and no token was provided, BYPASSING authorization check")
        return

    service_name = authz_config.service_name if authz_config else authz_service_name

    token = await _get_token(token, request)

    # either this was provided or we've tried to get it from the Bearer header
    if not token:
        raise HTTPException(status_code=HTTP_401_UNAUTHENTICATED)

    # try to get the ID so the debug log has more information
    try:
        user_id = await get_user_id(token, request, authz_config=authz_config)
    except HTTPException as exc:
        logging.debug(f"Unable to determine user_id. Defaulting to `Unknown`. Exc: {exc}")
        user_id = "Unknown"

    client = authz_config.arborist_client(request) if authz_config else arborist

    try:
        is_authorized = await client.auth_request(
            token.credentials,
            service=service_name,
            methods=authz_access_method,
            resources=authz_resources,
        )
    except ArboristError as exc:
        # A malformed or unverifiable token surfaces here as an Arborist error rather than
        # during claim parsing, so it is the caller's problem, not the service's.
        logging.error(f"ArboristError during auth_request: {exc}", exc_info=True)
        raise HTTPException(
            status_code=HTTP_401_UNAUTHENTICATED,
            detail="Could not verify, parse, and/or validate the provided access token.",
        ) from exc
    except Exception as exc:
        logging.error(f"arborist.auth_request failed, exc: {exc}", exc_info=True)
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authorization service error while checking access policies.",
        ) from exc

    if not is_authorized:
        logging.debug(
            f"user `{user_id}` does not have `{authz_access_method}` access "
            f"on `{authz_resources}` for service `{service_name}`"
        )
        raise HTTPException(status_code=HTTP_403_FORBIDDEN)


async def get_user_id(
    token: HTTPAuthorizationCredentials | None = None,
    request: Request | None = None,
    authz_config: AuthzConfig | None = None,
):
    """
    Retrieves the user ID from the provided token/request

    Args:
        token (HTTPAuthorizationCredentials): an authorization token (optional, you can also provide request
            and this can be parsed from there). this has priority over any token from request.
        request (Request): The incoming HTTP request. Used to parse tokens from header.
        authz_config (AuthzConfig | None): Service authz settings, used for the expected
            token audience.

    Returns:
        str: The user's ID.

    Raises:
        HTTPException: Raised if the token is missing or invalid.

    Note:
        If `DEBUG_SKIP_AUTH` is enabled and no token is provided, user_id is set to "0".
    """
    if config.DEBUG_SKIP_AUTH and not token:
        logging.warning("DEBUG_SKIP_AUTH mode is on and no token was provided, RETURNING user_id = 0")
        return "0"

    cached = _cache_get(request, "user_id")
    if cached is not None:
        return cached

    token_claims = await _get_token_claims(token, request, authz_config=authz_config)
    if "sub" not in token_claims:
        raise HTTPException(status_code=HTTP_401_UNAUTHENTICATED)

    user_id = token_claims["sub"]
    _cache_set(request, "user_id", user_id)
    return user_id


async def get_username(
    token: HTTPAuthorizationCredentials | None = None,
    request: Request | None = None,
    authz_config: AuthzConfig | None = None,
) -> str:
    """
    Retrieves the username from the provided token/request

    Args:
        token (HTTPAuthorizationCredentials): an authorization token (optional, you can also provide request
            and this can be parsed from there). this has priority over any token from request.
        request (Request): The incoming HTTP request. Used to parse tokens from header.
        authz_config (AuthzConfig | None): Service authz settings, used for the expected
            token audience.

    Returns:
        str: The user's username.

    Raises:
        HTTPException: Raised if the token is missing or invalid.

    Note:
        If `DEBUG_SKIP_AUTH` is enabled and no token is provided, username is set to "librarian".
    """
    if config.DEBUG_SKIP_AUTH and not token:
        logging.warning("DEBUG_SKIP_AUTH mode is on and no token was provided, RETURNING username = 'librarian'")
        return "0"

    token_claims = await _get_token_claims(token, request, authz_config=authz_config)
    if "user" not in token_claims.get("context", {}):
        raise HTTPException(status_code=HTTP_401_UNAUTHENTICATED)

    username = token_claims["context"]["user"]["name"]
    return username


async def get_user_authz_mapping(
    token: HTTPAuthorizationCredentials | None = None,
    request: Request | None = None,
    authz_config: AuthzConfig | None = None,
) -> Mapping:
    """
    Retrieve the caller's full authorization mapping from the Gen3 policy engine.

    This is the "what can this user do anywhere" question, as opposed to
    `authorize_request`'s "may this user do X to Y". Services need it to scope a query to
    everything the caller may see, which cannot be expressed as a yes/no check.

    The result is cached on the request, since resolving several access methods for one
    request would otherwise repeat the call.

    Args:
        token (HTTPAuthorizationCredentials | None): The HTTP bearer token supplied in the
            Authorization header.
        request (Request | None): The FastAPI request, used for token parsing and caching.
        authz_config (AuthzConfig | None): Service authz settings, used to resolve the
            ArboristClient.

    Returns:
        Mapping: The authorization mapping, or an empty mapping if DEBUG_SKIP_AUTH is
        enabled and no token is provided.

    Raises:
        HTTPException: 401 if no valid token is present, 500 if the policy engine errors.
    """
    if config.DEBUG_SKIP_AUTH and not token:
        logging.warning("DEBUG_SKIP_AUTH mode is on and no token was provided, RETURNING no authz mapping")
        return {}

    cached = _cache_get(request, "user_authz_mapping")
    if cached is not None:
        return cached

    token = await _get_token(token, request)

    # either this was provided or we've tried to get it from the Bearer header
    if not token:
        raise HTTPException(status_code=HTTP_401_UNAUTHENTICATED)

    logging.debug("Got user's token. Using it to get authz mapping...")

    client = authz_config.arborist_client(request) if authz_config else arborist

    try:
        authz_mapping = await client.auth_mapping(jwt=token.credentials)
    except ArboristError as exc:
        logging.error(f"ArboristError while retrieving authz mapping: {exc}", exc_info=True)
        raise HTTPException(
            status_code=HTTP_401_UNAUTHENTICATED,
            detail="Could not verify, parse, and/or validate the provided access token.",
        ) from exc
    except Exception as exc:
        logging.error(f"Unexpected error while retrieving authz mapping from Arborist: {exc}", exc_info=True)
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authorization service error while retrieving access policies.",
        ) from exc

    logging.debug(f"Got user's authz mapping: {authz_mapping}")
    _cache_set(request, "user_authz_mapping", authz_mapping)
    return authz_mapping


def get_allowed_authz_from_mapping(
    authz_mapping: Mapping,
    method: str,
    service: str,
) -> list[str]:
    """
    Reduce an authz mapping to the resource paths the caller holds `method` on.

    The mapping looks like:
      {
         "/some/resource": [
            {"service": "gen3-embeddings", "method": "read"},
            {"service": "gen3-embeddings", "method": "create"},
         ],
         ...
      }

    A `"*"` in either field is a wildcard grant and matches.

    Args:
        authz_mapping (Mapping): Mapping as returned by the policy engine.
        method (str): The logical CRUD action, e.g. "read", "create", "update", "delete".
        service (str): The Arborist service name to filter on.

    Returns:
        list[str]: Matching resource paths. Empty is a valid fail-closed result meaning "no
        resources", not "all resources".
    """
    allowed: list[str] = []

    for resource, perms in authz_mapping.items():
        if not isinstance(perms, list):
            continue
        for entry in perms:
            if not isinstance(entry, dict):
                continue
            entry_service = entry.get("service")
            entry_method = entry.get("method")
            if (entry_service in {service, "*"}) and (entry_method in {method, "*"}):
                allowed.append(resource)
                break

    return allowed


async def get_allowed_authz_for_request(
    request: Request,
    method: str,
    authz_config: AuthzConfig,
) -> list[str]:
    """
    Return the resource paths this caller may act on with a specific access method.

    Cached per (request, method), since a single request may need more than one method's
    resource set and each miss costs a call to the policy engine.

    Args:
        request (Request): The incoming request.
        method (str): Access method to filter by, e.g. "read", "create", "update", "delete".
        authz_config (AuthzConfig): Service authz settings.

    Returns:
        list[str]: Authz resource paths the caller holds `method` on. Empty if none, which
        is a valid fail-closed result.

    Raises:
        HTTPException: 401 if no valid token is present, 500 if the policy engine errors.
    """
    cache_key = f"allowed_authz_{method}"
    cached = _cache_get(request, cache_key)
    if cached is not None:
        logging.debug(f"allowed_authz for {method} fetched from request cache")
        return cached

    user_authz_mapping = await get_user_authz_mapping(request=request, authz_config=authz_config)
    allowed_authz = get_allowed_authz_from_mapping(
        authz_mapping=user_authz_mapping,
        method=method,
        service=authz_config.service_name,
    )
    _cache_set(request, cache_key, allowed_authz)
    logging.debug(f"allowed_authz for {method}: {allowed_authz}")
    return allowed_authz


async def _get_token_claims(
    token: HTTPAuthorizationCredentials | None = None,
    request: Request | None = None,
    authz_config: AuthzConfig | None = None,
):
    """
    Retrieves and validates token claims from the provided token.

    Args:
        token (HTTPAuthorizationCredentials): an authorization token (optional, you can also provide request
            and this can be parsed from there). this has priority over any token from request.
        request (Request): The incoming HTTP request. Used to parse tokens from header.
        authz_config (AuthzConfig | None): Service authz settings, used for the expected
            audience.

    Returns:
        dict: The token claims.

    Raises:
        HTTPException: Raised if the token is missing or invalid.
    """
    cached = _cache_get(request, "token_claims")
    if cached is not None:
        return cached

    token = await _get_token(token, request)
    # either this was provided or we've tried to get it from the Bearer header
    if not token:
        raise HTTPException(status_code=HTTP_401_UNAUTHENTICATED)

    audience = authz_config.expected_audience(request) if authz_config else AuthzConfig("").expected_audience(request)

    try:
        # NOTE: token can be None if no Authorization header was provided, we expect
        #       this to cause a downstream exception since it is invalid
        logging.debug(f"checking access token for scopes: `user` and `openid` and audience: `{audience}`")
        token_claims = await access_token("user", "openid", audience=audience, purpose="access")(token)
    except Exception as exc:
        logging.error(exc.detail if hasattr(exc, "detail") else exc, exc_info=True)
        raise HTTPException(
            HTTP_401_UNAUTHENTICATED,
            "Could not verify, parse, and/or validate scope from provided access token.",
        ) from exc

    _cache_set(request, "token_claims", token_claims)
    return token_claims


async def _get_token(token, request):
    """
    Retrieves the token from the request's Bearer header or if there's no request, returns token

    Args:
        token (HTTPAuthorizationCredentials): The provided token, if available.
        request (Request): The incoming HTTP request.

    Returns:
        The obtained token.
    """
    if token:
        return token

    if not request:
        return token

    cached = _cache_get(request, "bearer_token")
    if cached is not None:
        return cached

    token = await get_bearer_token(request)
    _cache_set(request, "bearer_token", token)
    return token
