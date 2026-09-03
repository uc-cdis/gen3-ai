"""
FastAPI app creation, general entrypoint into the service.
"""

from contextlib import asynccontextmanager
from importlib.metadata import version

from fastapi import APIRouter, FastAPI, HTTPException, Request
from gen3authz.client.arborist.async_client import ArboristClient

from common.auth import get_user_id
from common.logging_setup import configure_logging
from common.metrics import ServiceMetrics, get_metrics_client
from common.telemetry import configure_tracing, instrument_class
from gen3_embeddings import config
from gen3_embeddings.config import logging
from gen3_embeddings.database.db import DataAccessLayer, close_pool, get_pool
from gen3_embeddings.routes.basic import basic_router
from gen3_embeddings.routes.collections import collections_router
from gen3_embeddings.routes.embeddings import embeddings_router
from gen3_embeddings.routes.search import vectorstore_search_router

API_REQUESTS_COUNTER = "gen3_embeddings_api_requests"
# Stands in for the path of a request that matched no route, so that scanners and typos
# share one time series instead of minting one per URL.
UNMATCHED_PATH = "<unmatched>"
# `/metrics` is a mounted sub-application rather than a route, so it has no template to label it
# with. Mirrors the mount in common/metrics.py.
METRICS_PATH = "/metrics"
API_REQUESTS_COUNTER_DESCRIPTION = "API requests for Gen3 Embeddings."

route_aggregator = APIRouter()
route_aggregator.include_router(embeddings_router)
route_aggregator.include_router(collections_router)
route_aggregator.include_router(vectorstore_search_router)
route_aggregator.include_router(basic_router)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Validate external dependencies at startup and hold them for the app's lifetime.

    Startup fails loudly rather than serving traffic against a database or policy engine
    the service cannot safely use.

    Args:
        app (FastAPI): The application being started.

    Raises:
        Exception: If the database is unreachable, the database role can bypass row-level
            security, or the policy engine is unhealthy.
    """
    # Startup logic
    await check_db_connection()

    logging.debug(f"Initializing Arborist ({config.ARBORIST_URL}) client for authorization...")
    app.state.arborist_client = ArboristClient(
        arborist_base_url=config.ARBORIST_URL,
    )
    if not config.DEBUG_SKIP_AUTH:
        await check_arborist_is_healthy(app)

    try:
        yield
    finally:
        await close_pool()


async def check_arborist_is_healthy(app):
    """
    Checks that we can talk to arborist

    Args:
        app (FastAPI): the fastapi app with arborist client

    Raises:
        Exception: If the policy engine reports itself unhealthy.
    """
    logging.debug("Startup policy engine (Arborist) connection test initiating...")
    arborist_client = app.state.arborist_client
    if not await arborist_client.healthy():
        logging.exception(
            "Startup policy engine (Arborist) connection test FAILED. Unable to connect to the policy engine."
        )
        logging.debug("Arborist is unhealthy")
        raise Exception("Arborist unhealthy, aborting...")
    logging.debug("Startup policy engine (Arborist) connection test PASSED.")


async def check_db_connection():
    """
    Simple check to ensure we can talk to the db (asyncpg pool test)
    and ensure we are NOT using a superuser or bypassrls role.

    When DEBUG_SKIP_AUTH is True, we skip enforcing those checks, but
    emit a warning if the DB user cannot bypass RLS (i.e., is neither
    SUPERUSER nor has BYPASSRLS).

    Raises:
        Exception: If the database is unreachable, or the configured role is SUPERUSER or
            has BYPASSRLS, either of which would defeat row-level security.
    """
    try:
        logging.debug("Startup database connection test initiating. Attempting a simple query...")
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute("SELECT 1;")

            # Safety: verify current role privileges
            row = await conn.fetchrow(
                """
                SELECT usesuper, usebypassrls, usename
                FROM pg_user
                WHERE usename = current_user;
                """
            )

            usesuper = row["usesuper"]
            usebypassrls = row["usebypassrls"]
            usename = row["usename"]

            # If DEBUG_SKIP_AUTH is enabled, we do NOT enforce the "no superuser/bypassrls"
            # requirement, but we still log what we see.
            if config.DEBUG_SKIP_AUTH:
                if not usesuper and not usebypassrls:
                    logging.warning(
                        "DEBUG_SKIP_AUTH is True, but DB user '%s' is neither SUPERUSER "
                        "nor has BYPASSRLS. This user cannot bypass RLS; "
                        "RLS will still be enforced at the DB level.",
                        usename,
                    )
                else:
                    logging.debug(
                        "DEBUG_SKIP_AUTH is True and DB user '%s' has privileges "
                        "(usesuper=%s, usebypassrls=%s) that can bypass RLS.",
                        usename,
                        usesuper,
                        usebypassrls,
                    )

                # Skip the hard failure in DEBUG_SKIP_AUTH mode.
                logging.debug("Skipping DB superuser/bypassrls enforcement because DEBUG_SKIP_AUTH is True.")
            else:
                # Normal enforcement when DEBUG_SKIP_AUTH is False.
                if usesuper:
                    logging.error(f"DB user '{usename}' is SUPERUSER. This is unsafe for RLS.")
                    raise Exception(
                        "Configured DB user is SUPERUSER, which bypasses REQUIRED row-level security. Aborting..."
                    )

                if usebypassrls:
                    logging.error(f"DB user '{usename}' has BYPASSRLS. This is unsafe for RLS.")
                    raise Exception(
                        "Configured DB user has BYPASSRLS, which bypasses REQUIRED row-level security. Aborting..."
                    )

        logging.debug("Startup database connection test PASSED.")
    except Exception as exc:
        logging.exception("Startup database connection test FAILED. Unable to connect to the configured database.")
        logging.debug(exc)
        raise


def get_app() -> FastAPI:
    """
    Return the web framework app object after adding routes

    Returns:
        FastAPI: The FastAPI app object
    """
    app = FastAPI(
        title="Gen3 Embeddings Service",
        version=version("gen3_embeddings"),
        debug=config.DEBUG,
        root_path=config.URL_PREFIX,
        lifespan=lifespan,
    )
    configure_logging()
    configure_tracing(app, "gen3_embeddings")

    # A span per data-access call, which sits between the request span and the asyncpg spans
    # and is the layer the library instrumentation cannot see. Kept to work done once per
    # request: the helpers in models/helpers.py run per row inside bulk loops, where a span
    # would cost more than the work it measures. auth.py carries its own @traced decorators,
    # because its callers bind its functions by name at import and never look them up here.
    instrument_class(DataAccessLayer)

    # Mounts /metrics on the app as a side effect, so it has to run before the app serves traffic.
    app.state.metrics = ServiceMetrics(metrics_client=get_metrics_client(app))

    # Endpoints FastAPI and Starlette serve without an API route: the docs, the spec, and the
    # mounted metrics app. None of them leave a route on the request scope, so the middleware has
    # to recognise them by path.
    unrouted_paths = frozenset(path for path in (app.docs_url, app.redoc_url, app.openapi_url, METRICS_PATH) if path)

    @app.middleware("http")
    async def middleware_record_api_metric(request: Request, call_next):
        """
        Count every request that reaches a metered endpoint.

        Args:
            request (Request): the incoming HTTP request.
            call_next (Callable): the rest of the middleware stack, called by FastAPI.

        Returns:
            Response: the response produced downstream, unchanged.
        """
        response = await call_next(request)

        path = _get_path_label_for_metrics(request, unrouted_paths)
        if path in config.ENDPOINTS_WITHOUT_METRICS:
            return response

        metrics = getattr(app.state, "metrics", None)
        if not metrics or not metrics.metrics_client:
            return response

        try:
            user_id = await get_user_id(request=request)
        except HTTPException as exc:
            logging.debug(f"Could not retrieve user_id. Error: '{exc}'. Setting user_id to 'Unknown' for metrics")
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

    app.include_router(route_aggregator)

    return app


def _get_path_label_for_metrics(request: Request, unrouted_paths: frozenset[str]) -> str:
    """
    Return the label to record a request's path under.

    Args:
        request (Request): A request that has already been routed.
        unrouted_paths (frozenset[str]): Paths served without an API route, which therefore have
            no template to be labelled with.

    Returns:
        str: The matched route's template, for example
            `/vectorstore/collections/{collection_name}`, one of `unrouted_paths`, or
            UNMATCHED_PATH. Never the request's own URL unmatched, whose path parameters would
            each become a separate Prometheus time series.
    """
    template = getattr(request.scope.get("route"), "path", None)
    if template:
        return template

    # Starlette moves a mount's prefix from the path into root_path before handing the request to
    # the sub-application, so `/metrics` arrives here as root_path="/metrics", path="/". Matching
    # both, and only against known paths, keeps `/metrics` recognised - it would otherwise fall
    # through to UNMATCHED_PATH, miss the ENDPOINTS_WITHOUT_METRICS check below, and count every
    # single Prometheus scrape - while still collapsing anything unrouted into one series
    for candidate in (request.url.path, request.scope.get("root_path", "")):
        if candidate in unrouted_paths:
            return candidate

    return UNMATCHED_PATH


app_instance = get_app()
