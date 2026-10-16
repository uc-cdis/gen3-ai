"""
FastAPI app creation, general entrypoint into the service.
"""

from contextlib import asynccontextmanager
from importlib.metadata import version

from fastapi import APIRouter, FastAPI, HTTPException, Request
from gen3authz.client.arborist.async_client import ArboristClient
from gen3pyutils.observability.constants import UNKNOWN_LABEL_VALUE

from common.auth import get_user_id
from common.logging_setup import configure_logging
from common.metrics import add_request_metrics_middleware, get_metrics_client
from common.profiling import configure_profiling
from common.telemetry import configure_tracing, instrument_class
from gen3_embeddings import config
from gen3_embeddings.config import logging
from gen3_embeddings.database.dal.base import DataAccessLayerBase
from gen3_embeddings.database.dal.reads import ReadMixin
from gen3_embeddings.database.dal.search import SearchMixin
from gen3_embeddings.database.dal.writes import WriteMixin
from gen3_embeddings.database.db import create_pool
from gen3_embeddings.database.index_discovery import discover_vector_indexes
from gen3_embeddings.error_handlers import register_error_handlers
from gen3_embeddings.limits import RequestSizeLimitMiddleware
from gen3_embeddings.routes.basic import basic_router
from gen3_embeddings.routes.collections import collections_router
from gen3_embeddings.routes.embeddings import embeddings_router
from gen3_embeddings.routes.embeddings_bulk import embeddings_bulk_router
from gen3_embeddings.routes.search import vectorstore_search_router

# Tables whose authz enforcement depends on row-level security. Verified at startup by
# check_rls_is_enabled, since nothing else would notice if RLS were turned off.
RLS_PROTECTED_TABLES = ("embeddings_vector", "embeddings_halfvec", "collections")

SERVICE_DESCRIPTION = """
An authorization-scoped vector store for Gen3.

Embeddings are grouped into **collections**. A collection fixes two things at creation that
cannot be changed afterwards: the number of `dimensions` every vector in it must have, and the
`vector_type` those vectors are stored at (`vector` for float32, `halfvec` for float16). Write
embeddings with `POST`, replace them with `PUT`, and find the nearest ones with search.

Every request is authorized against a resource path. A collection's own path is
`/vectorstore/collections/{collection_name}`, but embeddings may be stored under any path you
hold the action on -- pass `authz` on write to choose one. Reads return only what your grants
cover, so an empty result and "not authorized" are deliberately indistinguishable.

Vectors are returned as JSON float arrays by default. For large vectors, the bulk read
endpoints return them base64-encoded instead, which is substantially cheaper to transfer.

Only pre-computed vectors are accepted today. Fields that take raw text are present in the
schemas so text input can be added without a breaking change, but supplying text returns a 400.
"""

# Tag order here is the order Redoc renders sections in, so it doubles as the shape of the
# published reference: collections first, because nothing else works without one.
OPENAPI_TAGS = [
    {
        "name": "Vectorstore Collections",
        "description": "Create and manage the collections that embeddings are grouped into.",
    },
    {
        "name": "Embeddings",
        "description": "Read and write individual embeddings, as JSON float arrays.",
    },
    {
        "name": "Embeddings (Bulk Read)",
        "description": (
            "Read many embeddings at once by UUID, with vectors base64-encoded rather than "
            "rendered as JSON arrays. These use `POST` so the UUID list can travel in the request "
            "body, but they only read."
        ),
    },
    {
        "name": "Vectorstore Search",
        "description": "Find the embeddings nearest to a query vector, within or across collections.",
    },
    {
        "name": "Service Info",
        "description": "Service version and health. Intended for operators rather than API clients.",
    },
]
API_REQUESTS_COUNTER = "gen3_embeddings_api_requests"
API_REQUESTS_COUNTER_DESCRIPTION = "API requests for Gen3 Embeddings."

route_aggregator = APIRouter()
route_aggregator.include_router(embeddings_router)
route_aggregator.include_router(embeddings_bulk_router)
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
        RuntimeError: If the database role can bypass row-level security, row-level security
            is not in effect, or the policy engine is unhealthy. An unreachable database
            raises the driver's own error instead.
    """
    # Startup logic. The pool is created here and only here, so every request path reads the
    # one on the app state instead of racing to build its own.
    app.state.db_pool = await create_pool()
    try:
        logging.debug(f"Initializing Arborist ({config.ARBORIST_URL}) client for authorization...")
        app.state.arborist_client = ArboristClient(
            arborist_base_url=config.ARBORIST_URL,
        )

        await check_db_connection(app.state.db_pool)

        # Which vector indexes exist decides which SQL shape search emits, and the query has
        # to match an index's expression exactly or Postgres silently scans instead. Read once
        # here rather than per request; an index built after startup is picked up on restart.
        async with app.state.db_pool.acquire() as conn:
            app.state.vector_indexes = await discover_vector_indexes(conn)

        if not config.DEBUG_SKIP_AUTH:
            await check_arborist_is_healthy(app)

        yield
    finally:
        # Also runs when a startup check raised, so a failed startup does not leak the pool's
        # connections.
        await app.state.db_pool.close()
        app.state.db_pool = None


async def check_arborist_is_healthy(app):
    """
    Checks that we can talk to arborist

    Args:
        app (FastAPI): the fastapi app with arborist client

    Raises:
        RuntimeError: If arborist is not healthy.
    """
    logging.debug("Startup policy engine (Arborist) connection test initiating...")
    arborist_client = app.state.arborist_client
    if not await arborist_client.healthy():
        logging.exception(
            "Startup policy engine (Arborist) connection test FAILED. Unable to connect to the policy engine."
        )
        logging.debug("Arborist is unhealthy")
        raise RuntimeError("Arborist unhealthy, aborting...")
    logging.debug("Startup policy engine (Arborist) connection test PASSED.")


async def check_rls_is_enabled(conn):
    """
    Ensure row-level security is enabled AND forced on the tables that rely on it.

    The superuser/bypassrls checks only cover the ROLE side of RLS. They pass happily against
    a database where RLS itself has been switched off (a stray
    `ALTER TABLE ... DISABLE ROW LEVEL SECURITY`, or a migration that never ran), in which
    case every row is visible to every caller with no warning. We otherwise rely solely on
    the migrations for this, so verify it at startup rather than assuming.

    Both conditions are required:

    - ENABLED, or no policy is applied at all and every row is visible.
    - FORCED, because a table's owner bypasses its own policies. Nothing else checks table
      ownership, so without FORCE the service could silently see everything just by
      connecting as the role that owns these tables.

    Note that RLS enabled with NO policy is safe: Postgres then denies all rows by default.
    So `relrowsecurity` is the condition worth enforcing, not the policy count.

    Args:
        conn: An open asyncpg connection.

    Raises:
        RuntimeError: If a protected table is missing, has RLS disabled, or does not have RLS
            forced, unless DEBUG_SKIP_AUTH is True, in which case it is a warning.
    """
    rows = await conn.fetch(
        """
        SELECT c.relname, c.relrowsecurity, c.relforcerowsecurity
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public'
          AND c.relname = ANY($1::text[]);
        """,
        list(RLS_PROTECTED_TABLES),
    )
    by_table = {row["relname"]: row for row in rows}

    missing = [table for table in RLS_PROTECTED_TABLES if table not in by_table]
    disabled = sorted(table for table, row in by_table.items() if not row["relrowsecurity"])
    not_forced = sorted(table for table, row in by_table.items() if not row["relforcerowsecurity"])

    if not missing and not disabled and not not_forced:
        logging.debug(f"Startup row-level security check PASSED for: {', '.join(RLS_PROTECTED_TABLES)}")
        return

    problems = []
    if disabled:
        problems.append(f"row-level security is DISABLED on: {', '.join(disabled)}")
    if not_forced:
        problems.append(f"row-level security is not FORCED on: {', '.join(not_forced)}")
    if missing:
        problems.append(f"expected table(s) not found: {', '.join(missing)}")
    detail = "; ".join(problems)

    if config.DEBUG_SKIP_AUTH:
        logging.warning(f"DEBUG_SKIP_AUTH is True, so continuing despite a row-level security problem: {detail}")
        return

    logging.error(f"Row-level security check failed: {detail}")
    raise RuntimeError(f"REQUIRED row-level security is not in effect ({detail}). Aborting...")


async def check_db_connection(pool):
    """
    Simple check to ensure we can talk to the db (asyncpg pool test),
    ensure we are NOT using a superuser or bypassrls role, and ensure
    row-level security is enabled and forced on the tables that depend on it.

    When DEBUG_SKIP_AUTH is True the role check inverts: the DB user MUST be able to bypass
    RLS (SUPERUSER or BYPASSRLS). Skipping auth resolves every request to no grants, so under
    any other role every read comes back empty and every write is rejected.

    Args:
        pool: The pool created for this app, whose lifetime the caller owns.

    Raises:
        RuntimeError: If the DB user is SUPERUSER or has BYPASSRLS, or row-level security is
            not in effect. Under DEBUG_SKIP_AUTH, if the DB user is neither SUPERUSER nor has
            BYPASSRLS, and row-level security not being in effect only warns. A database that
            cannot be reached raises the driver's own error instead.
    """
    try:
        logging.debug("Startup database connection test initiating. Attempting a simple query...")
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

            # DEBUG_SKIP_AUTH resolves every request to no grants, so RLS would hide every row and
            # reject every write. A role that bypasses RLS is the only one that state works with.
            if config.DEBUG_SKIP_AUTH:
                if not usesuper and not usebypassrls:
                    logging.error(
                        f"DEBUG_SKIP_AUTH is True, but DB user '{usename}' is neither SUPERUSER nor has BYPASSRLS."
                    )
                    raise RuntimeError(
                        "DEBUG_SKIP_AUTH is True, but the configured DB user cannot bypass row-level security, "
                        "so every request would run with no grants: reads return nothing and writes are "
                        "rejected. Connect as a SUPERUSER or BYPASSRLS role, or set DEBUG_SKIP_AUTH=False. "
                        "Aborting..."
                    )

                logging.debug(
                    f"DEBUG_SKIP_AUTH is True and DB user '{usename}' has privileges "
                    f"(usesuper={usesuper}, usebypassrls={usebypassrls}) that can bypass RLS."
                )
            else:
                # Normal enforcement when DEBUG_SKIP_AUTH is False.
                if usesuper:
                    logging.error(f"DB user '{usename}' is SUPERUSER. This is unsafe for RLS.")
                    raise RuntimeError(
                        "Configured DB user is SUPERUSER, which bypasses REQUIRED row-level security. Aborting..."
                    )

                if usebypassrls:
                    logging.error(f"DB user '{usename}' has BYPASSRLS. This is unsafe for RLS.")
                    raise RuntimeError(
                        "Configured DB user has BYPASSRLS, which bypasses REQUIRED row-level security. Aborting..."
                    )

            # The role checks above say nothing about whether RLS is actually switched on.
            await check_rls_is_enabled(conn)

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
        summary="Store, read, and search vector embeddings, scoped to what the caller is authorized to see.",
        description=SERVICE_DESCRIPTION,
        version=version("gen3_embeddings"),
        openapi_tags=OPENAPI_TAGS,
        debug=config.DEBUG,
        root_path=config.URL_PREFIX,
        lifespan=lifespan,
    )
    register_error_handlers(app)

    # Outermost, because it is the only limit that can act before the request body is read.
    # Every other bound in this service is checked against parsed Python objects, and the
    # parse is itself the expensive step: a 1 GiB JSON array costs a 1 GiB read plus the
    # memory of the resulting list before any validator sees it.
    app.add_middleware(RequestSizeLimitMiddleware, max_body_bytes=config.MAX_REQUEST_BODY_BYTES)
    configure_logging()

    # Before configure_tracing, which only links spans to profiles if the agent is already up.
    # One agent per process, and dockerrun.bash runs a single uvicorn process per container, so
    # one agent per container. Adding `--workers` would not leave the children unprofiled -
    # uvicorn spawns them and each imports the app, so each starts its own agent - but they would
    # all push under the same application name and the same `pod` tag, merging their flamegraphs.
    configure_profiling(config.DEPLOYMENT_SERVICE_NAME)
    configure_tracing(app, config.DEPLOYMENT_SERVICE_NAME)

    # A span per data-access call, which sits between the request span and the asyncpg spans
    # and is the layer the library instrumentation cannot see. Kept to work done once per
    # request: the helpers in models/helpers.py run per row inside bulk loops, where a span
    # would cost more than the work it measures. common.auth carries its own @traced decorators,
    # because its callers bind its functions by name at import and never look them up here.
    #
    # `instrument_class` only wraps the methods a class defines itself, and DataAccessLayer
    # defines none: they live on its base and mixins, so each of those is instrumented.
    for dal_class in (DataAccessLayerBase, ReadMixin, WriteMixin, SearchMixin):
        instrument_class(dal_class)

    # Mounts /metrics on the app as a side effect, so it has to run before the app serves traffic.
    app.state.metrics = get_metrics_client(app)

    add_request_metrics_middleware(
        app,
        app.state.metrics,
        counter_name=API_REQUESTS_COUNTER,
        counter_description=API_REQUESTS_COUNTER_DESCRIPTION,
        excluded_paths=config.ENDPOINTS_WITHOUT_METRICS,
        extra_label_names=("user_id",),
        extra_labels=_user_id_label,
    )

    app.include_router(route_aggregator)

    return app


async def _user_id_label(request: Request) -> dict[str, str]:
    """
    Return the user to attribute one counted request to.

    Args:
        request (Request): The request that was served.

    Returns:
        dict[str, str]: The `user_id` label. UNKNOWN_LABEL_VALUE for a request that carried no
            usable token, so an unauthenticated request is still counted.
    """
    try:
        return {"user_id": await get_user_id(request=request)}
    except HTTPException as exc:
        logging.debug(f"Could not retrieve user_id. Error: '{exc}'. Setting user_id to 'Unknown' for metrics")
        return {"user_id": UNKNOWN_LABEL_VALUE}


app_instance = get_app()
