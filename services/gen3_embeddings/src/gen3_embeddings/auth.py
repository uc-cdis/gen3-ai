from collections.abc import Mapping
from typing import Any

from authutils.token.fastapi import access_token
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from gen3authz.client.arborist.async_client import ArboristClient
from starlette.requests import Request
from starlette.status import HTTP_401_UNAUTHORIZED as HTTP_401_UNAUTHENTICATED
from starlette.status import HTTP_403_FORBIDDEN

from gen3_embeddings import config
from gen3_embeddings.config import logging

get_bearer_token = HTTPBearer(auto_error=False)


async def authorize_request(
    authz_access_method: str,
    authz_resources: list[str] | None = None,
    token: HTTPAuthorizationCredentials | None = None,
    request: Request | None = None,
):
    """
    Authorizes the incoming request based on the provided token and Arborist access policies.

    Args:
        authz_access_method (str): The Arborist access method to check (default is "access").
        authz_resources (list[str]): The list of resources to check against
        token (HTTPAuthorizationCredentials): an authorization token (optional, you can also provide request
            and this can be parsed from there). this has priority over any token from request.
        request: The incoming HTTP request. Used to parse tokens from header.

    Raises:
        HTTPException: Raised if authorization fails.

    Note:
        If `DEBUG_SKIP_AUTH` is enabled
        and no token is provided, the check is also bypassed.
    """
    if config.DEBUG_SKIP_AUTH and not token:
        logging.warning("DEBUG_SKIP_AUTH mode is on and no token was provided, BYPASSING authorization check")
        return

    token = await _get_token(token, request)

    # either this was provided or we've tried to get it from the Bearer header
    if not token:
        raise HTTPException(status_code=HTTP_401_UNAUTHENTICATED)

    # try to get the ID so the debug log has more information
    try:
        user_id = await get_user_id(token, request)
    except HTTPException as exc:
        logging.info(f"Unable to determine user_id. Defaulting to `Unknown`. Exc: {exc}")
        user_id = "Unknown"

    arborist_client = _get_arborist_client(request)

    is_authorized = await arborist_client.auth_request(
        token.credentials,
        service=config.AUTHZ_SERVICE_NAME,
        methods=authz_access_method,
        resources=authz_resources,
    )

    if not is_authorized:
        logging.info(f"user `{user_id}` does not have `{authz_access_method}` access on `{authz_resources}`")
        raise HTTPException(status_code=HTTP_403_FORBIDDEN)


async def get_user_id(token: HTTPAuthorizationCredentials | None = None, request: Request | None = None) -> int | Any:
    """
    Retrieves the user ID from the provided token/request

    Args:
        token (HTTPAuthorizationCredentials): an authorization token (optional, you can also provide request
            and this can be parsed from there). this has priority over any token from request.
        request: The incoming HTTP request. Used to parse tokens from header.

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

    token_claims = await _get_token_claims(token, request)
    if "sub" not in token_claims:
        raise HTTPException(status_code=HTTP_401_UNAUTHENTICATED)

    return token_claims["sub"]


async def get_user_authz_mapping(
    token: HTTPAuthorizationCredentials | None = None, request: Request | None = None
) -> Mapping:
    """
    Retrieve the user authorization mapping from the Gen3 Policy Engine

    In DEBUG_SKIP_AUTH mode and when no token is provided, this function
    returns an empty mapping instead of performing a call to Gen3 Policy Engine.

    Args:
        token (HTTPAuthorizationCredentials | None): The HTTP
            bearer token supplied in the Authorization header
        request (Request | None): The FastAPI request object, used to
            get the Gen3 Policy Engine client

    Returns:
        Mapping: The authorization mapping returned by the Gen3 Policy Engine client,
        or an empty mapping if DEBUG_SKIP_AUTH is enabled and no token is
        provided
    """
    if config.DEBUG_SKIP_AUTH and not token:
        logging.warning("DEBUG_SKIP_AUTH mode is on and no token was provided, RETURNING no authz mapping")
        return {}

    token = await _get_token(token, request)

    # either this was provided or we've tried to get it from the Bearer header
    if not token:
        raise HTTPException(status_code=HTTP_401_UNAUTHENTICATED)

    logging.debug("Got user's token. Using it to get authz mapping...")

    arborist_client = _get_arborist_client(request)

    authz_mapping = await arborist_client.auth_mapping(jwt=token.credentials)

    logging.debug(f"Got user's authz mapping: {authz_mapping}")

    return authz_mapping


async def _get_token_claims(
    token: HTTPAuthorizationCredentials | str | None = None,
    request: Request | None = None,
) -> dict:
    """
    Retrieves and validates token claims from the provided token.

    handler for proccessing token

    Args:
        token (HTTPAuthorizationCredentials): an authorization token (optional, you can also provide request
            and this can be parsed from there). this has priority over any token from request.
        request: The incoming HTTP request. Used to parse tokens from header.

    Returns:
        dict: The token claims.

    Raises:
        HTTPException: Raised if the token is missing or invalid.
    """
    token = await _get_token(token, request)
    # either this was provided or we've tried to get it from the Bearer header
    if not token:
        raise HTTPException(status_code=HTTP_401_UNAUTHENTICATED)

    # Audience is not used within Gen3 AuthN/Z service to individually represent Gen3 services
    # So don't bother setting it, b/c it doesn't add any additional security. Services
    # sometimes will add the URL of the Gen3 Auth service and look for that, but that does
    # nothing to validate themselves as recipients and we're already validating the signature.
    audience = None

    try:
        # NOTE: token can be None if no Authorization header was provided, we expect
        #       this to cause a downstream exception since it is invalid
        logging.debug(f"checking access token for scopes: `user` and `openid` and audience: `{audience}`")
        g = access_token("user", "openid", audience=audience, purpose="access")
        token_claims = await g(token)
    except Exception as exc:
        logging.error(exc.detail if hasattr(exc, "detail") else exc, exc_info=True)
        raise HTTPException(
            HTTP_401_UNAUTHENTICATED,
            "Could not verify, parse, and/or validate the provided access token.",
        ) from exc

    return token_claims


async def _get_token(token: HTTPAuthorizationCredentials | str | None, request: Request | None):
    """
    Retrieves the token from the request's Bearer header or if there's no request, returns token

    Args:
        token (HTTPAuthorizationCredentials): The provided token, if available.
        request: The incoming HTTP request.

    Returns:
        The obtained token.
    """
    if not token:
        # we need a request in order to get a bearer token
        if request:
            token = await get_bearer_token(request)
    return token


def _get_arborist_client(request: Request | None) -> ArboristClient:
    """
    Retrieves the Arborist client from the request's application state.

    This is primary broken out as a separate helper function to ease testing.

    Args:
        request: The incoming HTTP request containing the application state

    Returns:
        ArboristClient: The Arborist client instance from the application state
    """
    if not request:
        raise Exception("Expected a request, got None. Cannot determine Arborist Client from app state from request.")

    return request.app.state.arborist_client


def get_authz_resource_path_from_collection_name(collection_name: str) -> str:
    """
    Build the Arborist resource path for a vector collection.
    e.g. "/vector/indices/team42"
    """
    base = config.AUTHZ_SERVICE_RESOURCE.rstrip("/")
    if collection_name == "":
        return base
    else:
        return f"{base}/{collection_name}"


def get_allowed_authz_from_mapping(
    authz_mapping: Mapping,
    method: str,
    service: str | None = None,
) -> list[str]:
    """
    Given Arborist authz_mapping, return all resource paths for which the user has
    `method` access in `service`.

    - `method` is the logical CRUD action: "read", "create", "update", "delete"
    - `service` is typically config.AUTHZ_SERVICE_NAME ("gen3-embeddings").

    The mapping looks like:
      {
         "/vector/indices/collection_name": [
            {"service": "gen3-embeddings", "method": "read"},
            {"service": "gen3-embeddings", "method": "create"},
         ],
         ...
      }
    """
    service = service or config.AUTHZ_SERVICE_NAME
    allowed: list[str] = []

    for resource, perms in authz_mapping.items():
        if not isinstance(perms, list):
            continue
        for entry in perms:
            entry_service = entry.get("service")
            entry_method = entry.get("method")
            if (entry_service in {service, "*"}) and (entry_method in {method, "*"}):
                allowed.append(resource)
                break

    return allowed


def _get_crud_action_from_request(request: Request) -> str:
    """
    Return 'read', 'create', 'update', or 'delete' based on the HTTP verb.
    """
    method = request.method.upper()
    action = "unknown"

    if method == "GET":
        action = "read"
    if method == "POST":
        action = "create"
    if method in {"PUT", "PATCH"}:
        action = "update"
    if method == "DELETE":
        action = "delete"

    return action


async def parse_and_auth_request(request: Request, collection_name: str):
    """
    Authorize the request with arborist to ensure the request can be madefrom gen3_embeddings.auth import parse_and_auth_request

    Args:
        request: fastapi request entity

    Raises:
        HTTPException based on authorize_request outcome
    """
    user_id = await get_user_id(request=request)

    method = _get_crud_action_from_request(request=request)
    resource = get_authz_resource_path_from_collection_name(collection_name)

    logging.debug(f"Checking authorization for user: {user_id}. Method: {method}. Resource: {resource}.")
    await authorize_request(
        request=request,
        authz_access_method=method,
        authz_resources=[resource],
    )


async def get_allowed_authz_for_request(request: Request) -> list[str]:
    """
    Compute the allowed authz resource tags for this request, based on
    the user's authz mapping and the HTTP → CRUD mapping.

    This is used by the route layer to supply allowed_authz into the
    data access layer (DAL) for RLS.
    """
    user_authz_mapping = await get_user_authz_mapping(request=request)
    method = _get_crud_action_from_request(request)
    allowed_authz = get_allowed_authz_from_mapping(
        authz_mapping=user_authz_mapping,
        method=method,
    )
    logging.debug(f"allowed_authz for {method}: {allowed_authz}")
    return allowed_authz
