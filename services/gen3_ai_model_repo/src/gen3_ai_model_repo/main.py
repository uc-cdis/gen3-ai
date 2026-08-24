"""Gen3 AI Model Repo service application setup."""

from contextlib import asynccontextmanager
from importlib.metadata import version

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from gen3_ai_model_repo import config
from gen3_ai_model_repo.config import logging
from gen3_ai_model_repo.database.db import get_db_pool
from gen3_ai_model_repo.routes.router import route_aggregator
from gen3_ai_model_repo.storage.helpers import get_storage_provider


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    WIP

    Returns:
        FastAPI: The application, with no routes mounted yet.
    """
    logging.info("Starting up Gen3 AI Model Repository Service")

    await check_db_connection()
    await initialize_storage()

    yield


async def check_db_connection():
    """
    Simple check to ensure we can talk to the database.
    """

    try:
        logging.info("Startup database connection test initiating.")

        pool = await get_db_pool()

        async with pool.acquire() as conn:
            stmt = await conn.prepare("SELECT 1;")
            await stmt.fetchval()
        logging.info("Startup database connection test PASSED.")

    except Exception as exc:
        logging.exception("Startup database connection test FAILED.")
        logging.info(exc)
        raise


async def initialize_storage():
    """Initialize storage provider and ensure backing bucket/path exists."""
    try:
        provider = get_storage_provider()
        await provider.ensure_container()
        logging.info("Storage provider initialized and ready")
    except Exception:
        logging.exception("Startup storage initialization FAILED.")
        raise


def get_app() -> FastAPI:
    """
    Return configured FastAPI app.
    """

    app = FastAPI(
        title="Gen3 AI Model Repository Service",
        version=version("gen3_ai_model_repo"),
        debug=config.DEBUG,
        root_path=config.URL_PREFIX,
        lifespan=lifespan,
    )

    @app.get("/_status", include_in_schema=False)
    async def status():
        """Return the service liveness status for Kubernetes probes."""
        return {"status": "ok"}

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logging.exception(
            "Unhandled exception in model repository API",
            extra={"path": request.url.path, "method": request.method},
        )
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    app.include_router(route_aggregator)

    return app


app = get_app()
# Keep a stable gunicorn target used by deployment and `just run` recipes.
app_instance = app
