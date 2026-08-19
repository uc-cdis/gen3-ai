"""
Common router with general endpoints to be used the same way
across all Gen3 AI services.
"""

from fastapi import APIRouter
from fastapi.responses import RedirectResponse

common_router = APIRouter()


@common_router.get(
    "/",
    description="Directs client to the docs",
    summary="Get swagger docs",
    include_in_schema=False,
)
async def redirect_to_docs():
    """
    Redirects to the API docs if they hit the base endpoint.

    Returns:
        RedirectResponse: A redirect to /docs.
    """
    return RedirectResponse(url="/docs")
