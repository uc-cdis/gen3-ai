"""Route registration tests."""

import pytest
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
    monkeypatch.setattr(
        file_routes,
        "get_model_metadata",
        lambda *args: _async_value({"namespace": "ns", "repo": "repo", "description": "", "tags": []}),
    )
    monkeypatch.setattr(file_routes, "list_files_in_revision", lambda *args, **kwargs: _async_value([]))
    app = FastAPI()
    app.include_router(ai_models_files_router)
    app.dependency_overrides[verify_authorization] = lambda: None

    response_a = TestClient(app).get("/api/models/ns/repo/revision/main")
    response_b = TestClient(app).get("/api/models/ns/repo/revisions/main")

    assert response_a.status_code == 200
    assert response_b.status_code == 200
    body_a = response_a.json()
    body_b = response_b.json()
    assert body_a["id"] == body_b["id"] == "ns/repo"
    assert body_a["sha"] == body_b["sha"] == "commit-sha"
    assert body_a["metadata"] == {
        "namespace": "ns",
        "repo": "repo",
        "description": "",
        "tags": [],
        "created_at": None,
    }


def test_revision_list_and_detail_use_explicit_id_semantics(monkeypatch):
    """Revision detail uses the repo ID while the revision list keeps row IDs."""
    revision = {"id": 42, "revision": "main", "sha": "commit-sha", "etag": "etag"}
    monkeypatch.setattr(file_routes, "db_get_revision", lambda *args: _async_value(revision))
    monkeypatch.setattr(
        file_routes,
        "get_model_metadata",
        lambda *args: _async_value({"namespace": "ns", "repo": "repo", "description": "", "tags": []}),
    )
    monkeypatch.setattr(file_routes, "list_files_in_revision", lambda *args, **kwargs: _async_value([]))
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

    assert detail["id"] == "ns/repo"
    assert listing["id"] == "42"
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
    app.dependency_overrides[verify_authorization] = lambda: None

    client = TestClient(app)
    canonical_response = client.get("/api/models/ns/repo")
    alias_response = client.get("/api/models/ns/repo/info")
    assert canonical_response.status_code == 200
    assert alias_response.status_code == 200
    canonical = canonical_response.json()
    alias = alias_response.json()

    assert canonical["id"] == alias["id"] == "ns/repo"
    assert canonical["metadata"]["namespace"] == alias["metadata"]["namespace"] == "ns"
    assert canonical["metadata"]["repo"] == alias["metadata"]["repo"] == "repo"
    assert canonical["metadata"]["description"] == alias["metadata"]["description"] == "demo"
    assert canonical["metadata"]["tags"] == alias["metadata"]["tags"] == []
    assert alias == canonical


def test_repository_metadata_exposes_huggingface_siblings(monkeypatch):
    """Repository metadata includes tracked files in HF ``siblings`` form."""
    metadata = {"namespace": "ns", "repo": "repo", "description": "", "tags": []}
    revision = {"id": 42, "revision": "main", "sha": "commit-sha", "etag": "etag"}
    files = [{"path": "nested/config.json", "type": "file", "oid": "sha256", "size": 12}]
    monkeypatch.setattr(repo_routes, "get_model_metadata", lambda *args: _async_value(metadata))
    monkeypatch.setattr(repo_routes, "db_get_revision", lambda *args: _async_value(revision))
    monkeypatch.setattr(repo_routes, "list_files_in_revision", lambda *args, **kwargs: _async_value(files))
    app = FastAPI()
    app.include_router(ai_models_repositories_router)
    app.dependency_overrides[verify_authorization] = lambda: None

    response = TestClient(app).get("/api/models/ns/repo")

    assert response.status_code == 200
    assert response.json()["siblings"] == [
        {
            "rfilename": "nested/config.json",
            "size": 12,
            "blob_id": "sha256",
        },
    ]


def test_serialized_regular_siblings_parse_with_huggingface_model_info(monkeypatch):
    """Regular siblings parse with HF ModelInfo with or without file metadata."""
    huggingface_hub = pytest.importorskip("huggingface_hub")
    metadata = {"namespace": "ns", "repo": "repo", "description": "", "tags": []}
    revision = {"id": 42, "revision": "main", "sha": "commit-sha", "etag": "etag"}
    files = [{"path": "config.json", "type": "file", "oid": "sha256", "size": 12}]
    monkeypatch.setattr(repo_routes, "get_model_metadata", lambda *args: _async_value(metadata))
    monkeypatch.setattr(repo_routes, "db_get_revision", lambda *args: _async_value(revision))
    monkeypatch.setattr(repo_routes, "list_files_in_revision", lambda *args, **kwargs: _async_value(files))
    monkeypatch.setattr(file_routes, "get_model_metadata", lambda *args: _async_value(metadata))
    monkeypatch.setattr(file_routes, "db_get_revision", lambda *args: _async_value(revision))
    monkeypatch.setattr(file_routes, "list_files_in_revision", lambda *args, **kwargs: _async_value(files))
    app = FastAPI()
    app.include_router(ai_models_files_router)
    app.include_router(ai_models_repositories_router)
    app.dependency_overrides[verify_authorization] = lambda: None
    client = TestClient(app)

    response_bodies = [
        client.get("/api/models/ns/repo").json(),
        client.get("/api/models/ns/repo?files_metadata=true").json(),
        client.get("/api/models/ns/repo/revision/main").json(),
        client.get("/api/models/ns/repo/revision/main?files_metadata=true").json(),
    ]
    for body in response_bodies:
        parsed = huggingface_hub.ModelInfo(**body)
        assert parsed.siblings[0].lfs is None
        parsed_with_metadata = huggingface_hub.ModelInfo(**body)
        assert parsed_with_metadata.siblings[0].lfs is None


def test_huggingface_create_repository_contract(monkeypatch):
    """Create-repository requests create metadata and return the Hub identifier."""
    revisions = []
    monkeypatch.setattr(repo_routes, "db_model_exists", lambda *args: _async_value(False))

    async def create_model_with_revision(*args, **kwargs):
        revisions.append((args, kwargs))

    monkeypatch.setattr(repo_routes, "create_model_with_initial_revision", create_model_with_revision)
    app = FastAPI()
    app.include_router(ai_models_repositories_router)
    app.dependency_overrides[verify_authorization] = lambda: None

    response = TestClient(app).post("/api/repos/create", json={"name": "ns/repo", "repo_type": "model"})

    assert response.status_code == 200
    assert response.json() == {"url": "http://testserver/ns/repo", "repo_id": "ns/repo"}
    assert revisions[0][0] == ("ns", "repo")


def test_huggingface_preupload_selects_regular_mode():
    """The commit protocol receives an explicit regular-file upload mode."""
    app = FastAPI()
    app.include_router(ai_models_uploads_router)
    app.dependency_overrides[verify_authorization] = lambda: None

    response = TestClient(app).post(
        "/api/models/ns/repo/preupload/main",
        json={"files": [{"path": "nested/config.json", "sample": "YQ==", "size": 1}]},
    )

    assert response.status_code == 200
    assert response.json() == {
        "files": [{"path": "nested/config.json", "uploadMode": "regular", "shouldIgnore": False, "oid": None}]
    }


def _async_value(value):
    async def resolve():
        return value

    return resolve()
