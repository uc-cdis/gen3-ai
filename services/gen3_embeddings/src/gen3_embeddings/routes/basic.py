import time
from importlib.metadata import version

from fastapi import APIRouter
from starlette import status
from starlette.responses import JSONResponse, RedirectResponse

basic_router = APIRouter()


@basic_router.get(
    "/",
    description="Directs client to the docs",
    summary="Get swagger docs",
)
async def redirect_to_docs():
    return RedirectResponse(url="/docs")


@basic_router.get(
    "/_version/",
    status_code=status.HTTP_200_OK,
    description="Gets the current version of the service",
    summary="Get current version",
)
@basic_router.get("/_version", include_in_schema=False, dependencies=[])
async def get_version() -> dict:
    return {"version": version("gen3_embeddings")}


@basic_router.get(
    "/_status/",
    dependencies=[],
    description="Gets the current status of the service",
    summary="Get service status",
    responses={
        status.HTTP_200_OK: {"description": "Service is healthy"},
    },
)
@basic_router.get("/_status", include_in_schema=False, dependencies=[])
async def get_status() -> JSONResponse:
    """
    Very lightweight liveness/readiness endpoint.

    Returns 200 if the process is running and routes are mounted.
    Startup already validates DB and Arborist.
    """
    response = {"status": "OK", "timestamp": time.time()}
    return JSONResponse(status_code=status.HTTP_200_OK, content=response)
