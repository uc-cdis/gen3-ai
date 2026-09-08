"""Authorization helpers for the Gen3 AI model repo service."""

from fastapi import APIRouter, Request

from common.auth import authorize_request
from gen3_ai_model_repo import config

AUTHORIZATION_RESPONSES = {
    401: {"description": "User unauthenticated"},
    403: {"description": "User does not have access"},
}


class AuthorizedRouter(APIRouter):
    """Router that documents authentication failures for every route."""

    def add_api_route(self, path, endpoint, *, responses=None, **kwargs):
        """
        Add a route with standard authentication error responses.

        Returns:
            The registered API route.
        """
        documented_responses = {**AUTHORIZATION_RESPONSES, **(responses or {})}
        return super().add_api_route(path, endpoint, responses=documented_responses, **kwargs)


async def verify_authorization(request: Request):
    """
    FastAPI dependency for authentication and authorization.

    Validates a real bearer token (JWT), checks service-level access, and then
    checks repository-level CRUD authorization when the route targets a model.
    """

    # Keep the service-level permission as the first gate for all requests.
    await authorize_request(
        authz_resources=[config.AUTHZ_SERVICE_RESOURCE],
        authz_service_name=config.AUTHZ_SERVICE_NAME,
        authz_access_method="access",
        request=request,
    )

    # Model routes add a second, repository-level gate.  Routes without these
    # path parameters (for example, GET /api/models) are covered by the
    # service-level check only.
    path_params = request.path_params
    namespace = path_params.get("namespace")
    repo = path_params.get("repo")
    if namespace is None or repo is None:
        return

    method = _get_crud_action_from_request(request)
    resource = f"/ai_model_repo/{namespace}/{repo}"
    await authorize_request(
        authz_resources=[resource],
        authz_service_name=config.AUTHZ_SERVICE_NAME,
        authz_access_method=method,
        request=request,
    )


def _get_crud_action_from_request(request: Request) -> str:
    """
    Map an HTTP method to the CRUD action used by Arborist.

    Returns:
        The CRUD action corresponding to the request method.
    """
    method = request.method.upper()
    if method in {"GET", "HEAD"}:
        return "read"
    if method == "POST":
        return "create"
    if method in {"PUT", "PATCH"}:
        return "update"
    if method == "DELETE":
        return "delete"
    return "unknown"
