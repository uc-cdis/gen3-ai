from contextlib import asynccontextmanager
from importlib.metadata import version

from fastapi import APIRouter, FastAPI
from gen3authz.client.arborist.async_client import ArboristClient

from gen3_embeddings import config
from gen3_embeddings.config import logging
from gen3_embeddings.db import get_pool
from gen3_embeddings.routes.collections import collections_router
from gen3_embeddings.routes.embeddings import embeddings_router
from gen3_embeddings.routes.search import vectorstore_search_router

route_aggregator = APIRouter()
route_aggregator.include_router(embeddings_router)
route_aggregator.include_router(collections_router)
route_aggregator.include_router(vectorstore_search_router)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    await check_db_connection()

    app.state.arborist_client = ArboristClient(
        arborist_base_url=config.ARBORIST_URL,
    )
    if not config.DEBUG_SKIP_AUTH:
        await check_arborist_is_healthy(app)

    yield


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

            if usesuper:
                logging.error(f"DB user '{usename}' is SUPERUSER. This is unsafe for RLS.")
                raise Exception("Configured DB user is SUPERUSER, aborting...")

            if usebypassrls:
                logging.error(f"DB user '{usename}' has BYPASSRLS. This is unsafe for RLS.")
                raise Exception("Configured DB user has BYPASSRLS, aborting...")

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
