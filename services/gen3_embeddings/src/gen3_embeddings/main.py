from contextlib import asynccontextmanager
from importlib.metadata import version

from fastapi import APIRouter, FastAPI

from gen3_embeddings import config
from gen3_embeddings.config import logging
from gen3_embeddings.db import get_pool
from gen3_embeddings.routes.embeddings import embeddings_router

route_aggregator = APIRouter()
route_aggregator.include_router(embeddings_router, tags=["Embeddings"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    await check_db_connection()
    yield


async def check_db_connection():
    """
    Simple check to ensure we can talk to the db (asyncpg pool test)
    """
    try:
        logging.debug("Startup database connection test initiating. Attempting a simple query...")
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute("SELECT 1;")
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
    fastapi_app = FastAPI(
        title="Gen3 Embeddings Service",
        version=version("gen3_embeddings"),
        debug=config.DEBUG,
        root_path=config.URL_PREFIX,
        lifespan=lifespan,
    )
    fastapi_app.include_router(route_aggregator)

    return fastapi_app


app_instance = get_app()
