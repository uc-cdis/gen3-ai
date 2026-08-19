from datetime import datetime

from fastapi import FastAPI
from fastapi.testclient import TestClient

from gen3_ai_model_repo.auth import verify_authorization
from gen3_ai_model_repo.routes.ai_models_files import ai_models_files_router
from gen3_ai_model_repo.routes.ai_models_repositories import ai_models_repositories_router


def _build_test_client() -> TestClient:
    async def _fake_auth_override():
        return None

    app = FastAPI()
    app.include_router(ai_models_files_router)
    app.include_router(ai_models_repositories_router)
    app.dependency_overrides[verify_authorization] = _fake_auth_override
    return TestClient(app)


def test_list_models_empty_returns_200(monkeypatch):
    import gen3_ai_model_repo.routes.ai_models_repositories as repo_routes

    async def fake_list_models(namespace=None, tags=None, search=None):
        del namespace, tags, search
        return []

    monkeypatch.setattr(repo_routes, "list_models", fake_list_models)

    client = _build_test_client()
    response = client.get("/api/models")

    assert response.status_code == 200
    assert response.json() == []


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


def test_tree_missing_repository_returns_404(monkeypatch):
    import gen3_ai_model_repo.routes.ai_models_files as file_routes

    async def fake_exists(namespace, repo):
        del namespace, repo
        return False

    monkeypatch.setattr(file_routes, "db_model_exists", fake_exists)

    client = _build_test_client()
    response = client.get("/api/models/ns/repo/tree/main")

    assert response.status_code == 404


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
