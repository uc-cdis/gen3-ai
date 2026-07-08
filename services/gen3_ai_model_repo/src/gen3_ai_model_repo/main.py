from contextlib import asynccontextmanager
from importlib.metadata import version

from fastapi import FastAPI

from gen3_ai_model_repo import config
from gen3_ai_model_repo.config import logging
from gen3_ai_model_repo.database.db import get_db_pool
from gen3_ai_model_repo.routes.router import route_aggregator


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handle application startup and shutdown lifecycle.
    """
    print("Starting up Gen3 AI Model Repository Service")

    await check_db_connection()

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

    app.include_router(route_aggregator)

    return app


app = get_app()
