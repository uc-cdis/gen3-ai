"""Route registration tests."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

import gen3_ai_model_repo.routes.ai_models_files as file_routes
import gen3_ai_model_repo.routes.ai_models_repositories as repo_routes
from gen3_ai_model_repo.auth import verify_authorization
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
    assert "patch" in paths["/api/models/{namespace}/{repo}"]


def test_huggingface_metadata_and_resolve_aliases_are_registered():
    """Keep the canonical metadata and both HF resolve URL forms available."""
    app = FastAPI()
    app.include_router(ai_models_files_router)
    app.include_router(ai_models_repositories_router)
    app.include_router(ai_models_uploads_router)
    paths = TestClient(app).get("/openapi.json").json()["paths"]

    assert "/api/models/{namespace}/{repo}/info" in paths
    assert "/api/models/{namespace}/{repo}/resolve/{rev}/{path}" in paths
    assert "/{namespace}/{repo}/resolve/{rev}/{path}" in paths
    assert "/api/models/{namespace}/{repo}/revision/{revision}" in paths
    assert "/api/models/{namespace}/{repo}/revisions/{revision}" in paths


def test_revision_detail_aliases_return_identical_bodies(monkeypatch):
    """The singular and plural revision detail aliases share one behavior."""
    revision = {"id": 42, "revision": "main", "sha": "commit-sha", "etag": "etag"}
    monkeypatch.setattr(file_routes, "db_get_revision", lambda *args: _async_value(revision))
    app = FastAPI()
    app.include_router(ai_models_files_router)
    app.dependency_overrides[verify_authorization] = lambda: None

    response_a = TestClient(app).get("/api/models/ns/repo/revision/main")
    response_b = TestClient(app).get("/api/models/ns/repo/revisions/main")

    assert response_a.status_code == 200
    assert response_b.status_code == 200
    body_a = response_a.json()
    body_b = response_b.json()
    assert body_a == body_b
    assert body_a["id"] == body_b["id"] == "42"


def test_revision_list_and_detail_use_database_row_id(monkeypatch):
    """Revision list and detail responses use the database row ID consistently."""
    revision = {"id": 42, "revision": "main", "sha": "commit-sha", "etag": "etag"}
    monkeypatch.setattr(file_routes, "db_get_revision", lambda *args: _async_value(revision))
    monkeypatch.setattr(repo_routes, "db_model_exists", lambda *args: _async_value(True))
    monkeypatch.setattr(repo_routes, "list_revisions", lambda *args: _async_value([revision]))
    app = FastAPI()
    app.include_router(ai_models_files_router)
    app.include_router(ai_models_repositories_router)
    app.dependency_overrides[verify_authorization] = lambda: None

    client = TestClient(app)
    detail_response = client.get("/api/models/ns/repo/revision/main")
    listing_response = client.get("/api/models/ns/repo/revisions")
    assert detail_response.status_code == 200
    assert listing_response.status_code == 200
    detail = detail_response.json()
    listing = listing_response.json()["revisions"][0]

    assert detail["id"] == listing["id"] == "42"
    assert detail["sha"] == listing["sha"] == "commit-sha"


def test_info_matches_canonical_repository_metadata(monkeypatch):
    """The legacy /info alias returns the canonical repository body."""
    metadata = {"namespace": "ns", "repo": "repo", "description": "demo", "tags": []}
    revision = {"id": 42, "revision": "main", "sha": "commit-sha", "etag": "etag"}
    files = [{"type": "file", "oid": "sha", "size": 3, "path": "a.txt"}]
    monkeypatch.setattr(repo_routes, "get_model_metadata", lambda *args: _async_value(metadata))
    monkeypatch.setattr(repo_routes, "db_model_exists", lambda *args: _async_value(True))
    monkeypatch.setattr(repo_routes, "db_get_revision", lambda *args: _async_value(revision))
    monkeypatch.setattr(repo_routes, "list_files_in_revision", lambda *args, **kwargs: _async_value(files))
    app = FastAPI()
    app.include_router(ai_models_repositories_router)

    client = TestClient(app)
    canonical = client.get("/api/models/ns/repo").json()
    alias = client.get("/api/models/ns/repo/info").json()

    assert alias == canonical


def _async_value(value):
    async def resolve():
        return value

    return resolve()
