"""
FastAPI app creation, general entrypoint into the service.
"""

from contextlib import asynccontextmanager
from importlib.metadata import version

from fastapi import APIRouter, FastAPI
from gen3authz.client.arborist.async_client import ArboristClient

from gen3_embeddings import config
from gen3_embeddings.config import logging
from gen3_embeddings.database.db import create_pool
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
        Exception: If the database is unreachable, the database role can bypass row-level
            security, or the policy engine is unhealthy.
    """
    # Startup logic. The pool is created here and only here, so every request path reads the
    # one on the app state instead of racing to build its own.
    app.state.db_pool = await create_pool()

    logging.debug(f"Initializing Arborist ({config.ARBORIST_URL}) client for authorization...")
    app.state.arborist_client = ArboristClient(
        arborist_base_url=config.ARBORIST_URL,
    )

    try:
        await check_db_connection(app.state.db_pool)

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
        app_with_setup (FastAPI): the fastapi app with arborist client
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
        Exception: If a protected table is missing, has RLS disabled, or does not have RLS
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
        logging.debug("Startup row-level security check PASSED for: %s", ", ".join(RLS_PROTECTED_TABLES))
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
        logging.warning(
            "DEBUG_SKIP_AUTH is True, so continuing despite a row-level security problem: %s",
            detail,
        )
        return

    logging.error(f"Row-level security check failed: {detail}")
    raise Exception(f"REQUIRED row-level security is not in effect ({detail}). Aborting...")


async def check_db_connection(pool):
    """
    Simple check to ensure we can talk to the db (asyncpg pool test),
    ensure we are NOT using a superuser or bypassrls role, and ensure
    row-level security is enabled and forced on the tables that depend on it.

    When DEBUG_SKIP_AUTH is True, we skip enforcing those checks, but
    emit a warning if the DB user cannot bypass RLS (i.e., is neither
    SUPERUSER nor has BYPASSRLS).

    Args:
        pool: The pool created for this app, whose lifetime the caller owns.
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

    app.include_router(route_aggregator)

    return app


app_instance = get_app()
