from importlib.metadata import version

from fastapi import FastAPI

from gen3_ai_model_repo.routes.ai_models import ai_models_router


def get_app():
    app = FastAPI(
        title="Gen3 AI Model Repository Service",
        version=version("gen3_ai_model_repo"),
        # debug=config.DEBUG,
        # root_path=config.URL_PREFIX,
        # lifespan=lifespan,
    )

    app.include_router(ai_models_router)
    return app


app = get_app()
