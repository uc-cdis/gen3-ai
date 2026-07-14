from fastapi import FastAPI
from fastapi.testclient import TestClient

from gen3_ai_model_repo.routes.ai_models_files import ai_models_files_router
from gen3_ai_model_repo.routes.ai_models_repositories import ai_models_repositories_router
from gen3_ai_model_repo.routes.ai_models_uploads import ai_models_uploads_router


def test_openapi_includes_model_routes():
    """Verify OpenAPI schema includes core model route paths."""
    app = FastAPI()
    app.include_router(ai_models_files_router)
    app.include_router(ai_models_repositories_router)
    app.include_router(ai_models_uploads_router)
    client = TestClient(app)
    openapi = client.get("/openapi.json").json()
    paths = openapi["paths"]
    assert "/api/models" in paths
    assert "/api/models/{namespace}/{repo}" in paths
    assert "/api/models/{namespace}/{repo}/revisions" in paths
