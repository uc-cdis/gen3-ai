from importlib.metadata import version

from fastapi import FastAPI


def get_app():
    app = FastAPI(
        title="Gen3 AI Model Repository Service",
        version=version("gen3_ai_model_repo"),
        # debug=config.DEBUG,
        # root_path=config.URL_PREFIX,
        # lifespan=lifespan,
    )
    return app
