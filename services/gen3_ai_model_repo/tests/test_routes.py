from fastapi import FastAPI
from fastapi.testclient import TestClient

from gen3_ai_model_repo.routes.ai_models import ai_models_router


def test_openapi_includes_models_routes():
    app = FastAPI()
    app.include_router(ai_models_router)
    client = TestClient(app)
    openapi = client.get("/openapi.json").json()
    paths = openapi["paths"]
    assert "/api/models" in paths
    assert "/api/models/{namespace}/{repo}" in paths
    assert "/api/models/{namespace}/{repo}/revisions" in paths
