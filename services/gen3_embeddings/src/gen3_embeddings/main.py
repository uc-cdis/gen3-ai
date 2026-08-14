"""
FastAPI app creation, general entrypoint into the service.
"""

from contextlib import asynccontextmanager
from importlib.metadata import version

from fastapi import APIRouter, FastAPI
from gen3authz.client.arborist.async_client import ArboristClient

from gen3_embeddings import config
from gen3_embeddings.config import logging
from gen3_embeddings.database.db import close_pool, get_pool
from gen3_embeddings.routes.basic import basic_router
from gen3_embeddings.routes.collections import collections_router
from gen3_embeddings.routes.embeddings import embeddings_router
from gen3_embeddings.routes.embeddings_bulk import embeddings_bulk_router
from gen3_embeddings.routes.search import vectorstore_search_router

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


async def check_db_connection():
    """
    Simple check to ensure we can talk to the db (asyncpg pool test)
    and ensure we are NOT using a superuser or bypassrls role.

    When DEBUG_SKIP_AUTH is True, we skip enforcing those checks, but
    emit a warning if the DB user cannot bypass RLS (i.e., is neither
    SUPERUSER nor has BYPASSRLS).
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
    app.include_router(route_aggregator)

    return app


app_instance = get_app()
