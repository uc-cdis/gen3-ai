"""Gen3 AI Model Repo service application setup."""

from contextlib import asynccontextmanager
from importlib.metadata import version

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from common.auth import get_user_id
from common.logging_setup import configure_logging
from common.metrics import get_metrics_client
from common.telemetry import configure_tracing
from gen3_ai_model_repo import config
from gen3_ai_model_repo.config import logging
from gen3_ai_model_repo.database.db import close_db, get_db_pool
from gen3_ai_model_repo.database.repo_metadata import get_repository_metrics
from gen3_ai_model_repo.metrics import AiModelRepoServiceMetrics
from gen3_ai_model_repo.routes.router import route_aggregator
from gen3_ai_model_repo.storage.helpers import get_storage_provider

API_REQUESTS_COUNTER = "gen3_ai_model_repo_api_requests"
API_REQUESTS_COUNTER_DESCRIPTION = "API requests for Gen3 AI Model Repo."
METRICS_PATH = "/metrics"
UNMATCHED_PATH = "<unmatched>"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.

    Handles startup and shutdown of the application.
    """
    logging.info("Starting up Gen3 AI Model Repository Service")

    await check_db_connection()
    await initialize_storage()
    await update_metrics(app)

    try:
        yield
    finally:
        await close_db()


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

    Returns:
        FastAPI: A fully configured FastAPI application instance.
    """

    app = FastAPI(
        title="Gen3 AI Model Repository Service",
        version=version("gen3_ai_model_repo"),
        debug=config.DEBUG,
        root_path=config.URL_PREFIX,
        lifespan=lifespan,
    )
    configure_logging()
    configure_tracing(app, "gen3_ai_model_repo")
    app.state.metrics = AiModelRepoServiceMetrics(metrics_client=get_metrics_client(app))

    unrouted_paths = frozenset(path for path in (app.docs_url, app.redoc_url, app.openapi_url, METRICS_PATH) if path)

    @app.middleware("http")
    async def middleware_record_api_metric(request: Request, call_next):
        response = await call_next(request)

        if request.method not in {"GET", "HEAD"}:
            try:
                await update_metrics(app)
            except Exception:
                # Metrics refresh must not change the result of a successful API mutation.
                logging.exception("Failed to refresh model repository metrics")

        path = _get_path_label_for_metrics(request, unrouted_paths)
        if path in config.ENDPOINTS_WITHOUT_METRICS:
            return response

        metrics = getattr(app.state, "metrics", None)
        if not metrics or not metrics.metrics_client:
            return response

        try:
            user_id = await get_user_id(request=request)
        except HTTPException:
            user_id = "Unknown"

        metrics.add_to_api_interaction_counter(
            name=API_REQUESTS_COUNTER,
            description=API_REQUESTS_COUNTER_DESCRIPTION,
            method=request.method,
            path=path,
            status_code=response.status_code,
            user_id=user_id,
        )

        return response

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


async def update_metrics(app: FastAPI) -> None:
    """Refresh repository gauges from the database after startup or mutations."""
    metrics = getattr(app.state, "metrics", None)
    if not metrics or not metrics.metrics_client or not metrics.metrics_client.enabled:
        return

    model_count, file_count, total_size_bytes = await get_repository_metrics()
    metrics.add_models_count_metric(model_count)
    metrics.add_stored_files_count_metric(file_count)
    metrics.add_stored_models_size_metric(total_size_bytes)


def _get_path_label_for_metrics(request: Request, unrouted_paths: frozenset[str]) -> str:
    """Return a bounded route label for the API request metric."""
    template = getattr(request.scope.get("route"), "path", None)
    if template:
        return template

    for candidate in (request.url.path, request.scope.get("root_path", "")):
        if candidate in unrouted_paths:
            return candidate

    return UNMATCHED_PATH


app = get_app()
# Keep a stable Uvicorn application target used by deployment and `just run` recipes.
app_instance = app
