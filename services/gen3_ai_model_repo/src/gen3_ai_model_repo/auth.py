"""Authorization helpers for the Gen3 AI model repo service."""

from fastapi import Request

from common.auth import authorize_request
from gen3_ai_model_repo import config


async def verify_authorization(request: Request):
    """
    FastAPI dependency for authentication and authorization.

    Validates a real bearer token (JWT) and checks Arborist authorization for
    this service-level resource.
    """

    await authorize_request(
        authz_resources=[config.AUTHZ_SERVICE_RESOURCE],
        authz_service_name=config.AUTHZ_SERVICE_NAME,
        authz_access_method="access",
        request=request,
    )
