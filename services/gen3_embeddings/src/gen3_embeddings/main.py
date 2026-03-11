from importlib.metadata import version

import asyncpg
from fastapi import FastAPI

from gen3_embeddings import config

print(asyncpg.__version__)


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
        # root_path=config.URL_PREFIX,
        # lifespan=lifespan,
    )

    return fastapi_app


app_instance = get_app()
