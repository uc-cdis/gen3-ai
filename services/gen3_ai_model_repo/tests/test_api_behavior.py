from datetime import datetime

from fastapi import FastAPI
from fastapi.testclient import TestClient

from gen3_ai_model_repo.auth import verify_authorization
from gen3_ai_model_repo.routes.ai_models_files import ai_models_files_router
from gen3_ai_model_repo.routes.ai_models_repositories import ai_models_repositories_router
from gen3_ai_model_repo.routes.ai_models_uploads import ai_models_uploads_router


def _build_test_client() -> TestClient:
    async def _fake_auth_override():
        return None

    app = FastAPI()
    app.include_router(ai_models_files_router)
    app.include_router(ai_models_repositories_router)
    app.include_router(ai_models_uploads_router)
    app.dependency_overrides[verify_authorization] = _fake_auth_override
    return TestClient(app)


def test_list_models_empty_returns_200(monkeypatch):
    import gen3_ai_model_repo.routes.ai_models_repositories as repo_routes

    async def fake_list_models(namespace=None, tags=None, search=None, limit=100, offset=0):
        del namespace, tags, search, limit, offset
        return []

    monkeypatch.setattr(repo_routes, "list_models", fake_list_models)

    client = _build_test_client()
    response = client.get("/api/models")

    assert response.status_code == 200
    assert response.json() == {"models": [], "page": 1, "page_size": 100, "next_page": None, "prev_page": None}


def test_list_model_revisions_empty_returns_200(monkeypatch):
    import gen3_ai_model_repo.routes.ai_models_repositories as repo_routes

    async def fake_exists(namespace, repo):
        del namespace, repo
        return True

    async def fake_list_revisions(namespace, repo):
        del namespace, repo
        return []

    monkeypatch.setattr(repo_routes, "db_model_exists", fake_exists)
    monkeypatch.setattr(repo_routes, "list_revisions", fake_list_revisions)

    client = _build_test_client()
    response = client.get("/api/models/ns/repo/revisions")

    assert response.status_code == 200
    assert response.json() == {"repo": "ns/repo", "revisions": []}


def test_tree_empty_repository_returns_200_empty_list(monkeypatch):
    import gen3_ai_model_repo.routes.ai_models_files as file_routes

    async def fake_exists(namespace, repo):
        del namespace, repo
        return True

    async def fake_list_files_in_revision(namespace, model_name, revision_name="main"):
        del namespace, model_name, revision_name
        return []

    monkeypatch.setattr(file_routes, "db_model_exists", fake_exists)
    monkeypatch.setattr(file_routes, "list_files_in_revision", fake_list_files_in_revision)

    client = _build_test_client()
    response = client.get("/api/models/ns/repo/tree/main")

    assert response.status_code == 200
    assert response.json() == []


def test_tree_path_filter_matches_path_segments(monkeypatch):
    import gen3_ai_model_repo.routes.ai_models_files as file_routes

    async def fake_exists(namespace, repo):
        del namespace, repo
        return True

    async def fake_list_files_in_revision(namespace, model_name, revision_name="main"):
        del namespace, model_name, revision_name
        return [
            {"path": "config.json", "type": "file", "oid": "config", "size": 1},
            {"path": "conf/settings.json", "type": "file", "oid": "settings", "size": 2},
            {"path": "config.yaml", "type": "file", "oid": "yaml", "size": 3},
        ]

    monkeypatch.setattr(file_routes, "db_model_exists", fake_exists)
    monkeypatch.setattr(file_routes, "list_files_in_revision", fake_list_files_in_revision)

    client = _build_test_client()
    response = client.get("/api/models/ns/repo/tree/main/conf")

    assert response.status_code == 200
    assert response.json() == [{"type": "file", "oid": "settings", "size": 2, "path": "conf/settings.json"}]


def test_tree_missing_repository_returns_404(monkeypatch):
    import gen3_ai_model_repo.routes.ai_models_files as file_routes

    async def fake_exists(namespace, repo):
        del namespace, repo
        return False

    monkeypatch.setattr(file_routes, "db_model_exists", fake_exists)

    client = _build_test_client()
    response = client.get("/api/models/ns/repo/tree/main")

    assert response.status_code == 404


def test_file_id_rejects_malformed_and_mismatched_ids():
    client = _build_test_client()

    malformed = client.get("/api/models/ns/repo/files/config.json")
    mismatched = client.delete("/api/models/ns/repo/files/other:repository:main:config.json")

    assert malformed.status_code == 422
    assert mismatched.status_code == 422


def test_update_repository_requires_authorization():
    app = FastAPI()
    app.include_router(ai_models_repositories_router)
    client = TestClient(app)

    response = client.patch("/api/models/ns/repo", json={"description": "updated"})

    assert response.status_code in {401, 403}


def test_upload_rejects_too_many_files(monkeypatch):
    import gen3_ai_model_repo.routes.ai_models_uploads as upload_routes

    monkeypatch.setattr(upload_routes.config, "MAX_UPLOAD_FILES", 1)
    client = _build_test_client()

    response = client.post(
        "/api/models/ns/repo/upload",
        files=[
            ("files", ("one.bin", b"1", "application/octet-stream")),
            ("files", ("two.bin", b"2", "application/octet-stream")),
        ],
    )

    assert response.status_code == 413


def test_upload_rejects_content_length_before_processing(monkeypatch):
    import gen3_ai_model_repo.routes.ai_models_uploads as upload_routes

    monkeypatch.setattr(upload_routes.config, "MAX_UPLOAD_BYTES", 1)
    client = _build_test_client()

    response = client.post(
        "/api/models/ns/repo/upload",
        files=[("files", ("model.bin", b"model", "application/octet-stream"))],
    )

    assert response.status_code == 413


def test_update_repository_metadata(monkeypatch):
    import gen3_ai_model_repo.routes.ai_models_repositories as repo_routes

    async def fake_exists(namespace, repo):
        del namespace, repo
        return True

    async def fake_update(namespace, model_name, description=None, tags=None):
        return {
            "namespace": namespace,
            "repo": model_name,
            "description": description,
            "tags": tags or [],
            "created_at": datetime(2026, 7, 21),
        }

    monkeypatch.setattr(repo_routes, "db_model_exists", fake_exists)
    monkeypatch.setattr(repo_routes, "update_model_metadata", fake_update)

    client = _build_test_client()
    response = client.patch(
        "/api/models/ns/repo",
        json={"description": "updated", "tags": ["prod"]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["namespace"] == "ns"
    assert body["repo"] == "repo"
    assert body["description"] == "updated"
    assert body["tags"] == ["prod"]
