"""API behavior tests for model repository routes."""

from datetime import datetime

import pytest
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
    """Return an empty paginated model list when the database has no models."""
    import gen3_ai_model_repo.routes.ai_models_repositories as repo_routes

    async def fake_list_models(
        namespace=None,
        tags=None,
        search=None,
        limit=100,
        offset=0,
        permitted_repositories=None,
        permitted_resource_paths=None,
    ):
        del namespace, tags, search, limit, offset, permitted_repositories, permitted_resource_paths
        return []

    monkeypatch.setattr(repo_routes, "list_models", fake_list_models)

    client = _build_test_client()
    response = client.get("/api/models")

    assert response.status_code == 200
    assert response.json() == {"models": [], "page": 1, "page_size": 100, "next_page": None, "prev_page": None}


def test_list_model_revisions_empty_returns_200(monkeypatch):
    """Return an empty revision list for an existing repository."""
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
    """Return an empty tree for an existing repository without files."""
    import gen3_ai_model_repo.routes.ai_models_files as file_routes

    async def fake_exists(namespace, repo):
        del namespace, repo
        return True

    async def fake_list_files_in_revision(namespace, model_name, revision_name="main"):
        del namespace, model_name, revision_name
        return []

    async def fake_get_revision(namespace, model_name, revision_name):
        del namespace, model_name
        return {"revision": revision_name, "sha": "commit-sha"}

    monkeypatch.setattr(file_routes, "db_model_exists", fake_exists)
    monkeypatch.setattr(file_routes, "list_files_in_revision", fake_list_files_in_revision)
    monkeypatch.setattr(file_routes, "db_get_revision", fake_get_revision)

    client = _build_test_client()
    response = client.get("/api/models/ns/repo/tree/main")

    assert response.status_code == 200
    assert response.json() == []


def test_remote_resolve_redirect_preserves_file_metadata_headers(monkeypatch):
    """GET and HEAD remote redirects carry revision and file identity metadata."""
    import gen3_ai_model_repo.routes.ai_models_files as file_routes

    content = b"remote-content"
    record = {
        "object_key": "ns/repo/main/config.json",
        "size": len(content),
        "sha": "revision-sha",
        "etag": "content-sha",
    }

    class Provider:
        async def get_file_path(self, object_key):
            assert object_key == record["object_key"]
            return None

        async def generate_signed_url(self, object_key):
            assert object_key == record["object_key"]
            return "http://storage.invalid/models/config.json?signature=redacted"

    async def get_file_record(*args, **kwargs):
        del args, kwargs
        return record

    monkeypatch.setattr(file_routes, "get_file_record", get_file_record)
    monkeypatch.setattr(file_routes, "get_storage_provider", lambda: Provider())
    client = _build_test_client()

    for method in ("get", "head"):
        response = client.request(method, "/api/models/ns/repo/resolve/main/config.json", follow_redirects=False)
        assert response.status_code == 302
        assert response.headers["location"].startswith("http://storage.invalid/")
        assert response.headers["x-repo-commit"] == "revision-sha"
        assert response.headers["x-linked-etag"] == "content-sha"
        assert response.headers["x-linked-size"] == str(len(content))
        assert response.headers.get("content-length") != str(len(content))


def test_tree_path_filter_matches_path_segments(monkeypatch):
    """Filter tree results by directory path segments."""
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

    async def fake_get_revision(namespace, model_name, revision_name):
        del namespace, model_name
        return {"revision": revision_name, "sha": "commit-sha"}

    monkeypatch.setattr(file_routes, "db_model_exists", fake_exists)
    monkeypatch.setattr(file_routes, "list_files_in_revision", fake_list_files_in_revision)
    monkeypatch.setattr(file_routes, "db_get_revision", fake_get_revision)

    client = _build_test_client()
    response = client.get("/api/models/ns/repo/tree/main/conf")

    assert response.status_code == 200
    assert response.json() == [{"type": "file", "oid": "settings", "size": 2, "path": "conf/settings.json"}]


def test_tree_recursive_resolves_branch_and_commit_with_nested_files(monkeypatch):
    """HF tree enumeration accepts both main and its returned commit SHA."""
    import gen3_ai_model_repo.routes.ai_models_files as file_routes

    files = [
        {"path": "config.json", "type": "file", "oid": "config", "size": 1},
        {"path": "1_Pooling/config.json", "type": "file", "oid": "pool", "size": 2},
    ]

    async def fake_exists(namespace, repo):
        del namespace, repo
        return True

    async def fake_revision(namespace, repo, revision):
        del namespace, repo
        if revision in {"main", "commit-sha"}:
            return {"revision": "main", "sha": "commit-sha"}
        return None

    async def fake_files(namespace, model_name, revision_name="main"):
        del namespace, model_name, revision_name
        return files

    monkeypatch.setattr(file_routes, "db_model_exists", fake_exists)
    monkeypatch.setattr(file_routes, "db_get_revision", fake_revision)
    monkeypatch.setattr(file_routes, "list_files_in_revision", fake_files)
    client = _build_test_client()

    for revision in ("main", "commit-sha"):
        response = client.get(f"/api/models/ns/repo/tree/{revision}?recursive=true")
        assert response.status_code == 200
        assert {entry["path"] for entry in response.json()} == {"config.json", "1_Pooling/config.json"}


def test_tree_missing_repository_returns_404(monkeypatch):
    """Return not found when the requested repository does not exist."""
    import gen3_ai_model_repo.routes.ai_models_files as file_routes

    async def fake_exists(namespace, repo):
        del namespace, repo
        return False

    monkeypatch.setattr(file_routes, "db_model_exists", fake_exists)

    client = _build_test_client()
    response = client.get("/api/models/ns/repo/tree/main")

    assert response.status_code == 404


def test_file_id_rejects_malformed_and_mismatched_ids():
    """Reject malformed and cross-repository file identifiers."""
    client = _build_test_client()

    malformed = client.get("/api/models/ns/repo/files/config.json")
    mismatched = client.delete("/api/models/ns/repo/files/other:repository:main:config.json")

    assert malformed.status_code == 422
    assert mismatched.status_code == 422


def test_update_repository_requires_authorization(monkeypatch):
    """Require authorization for repository metadata updates."""
    import gen3_ai_model_repo.auth as auth

    async def deny_request(**kwargs):
        from fastapi import HTTPException

        raise HTTPException(status_code=403, detail="denied")

    monkeypatch.setattr(auth, "authorize_request", deny_request)
    app = FastAPI()
    app.include_router(ai_models_repositories_router)
    client = TestClient(app)

    response = client.patch("/api/models/ns/repo", json={"description": "updated"})

    assert "/api/models/{namespace}/{repo}" in client.get("/openapi.json").json()["paths"]
    assert response.status_code == 403


@pytest.mark.parametrize(
    ("method", "path", "expected_action"),
    [
        ("GET", "/api/models/ns/repo", "read"),
        ("HEAD", "/api/models/ns/repo/resolve/main/config.json", "read"),
        ("PATCH", "/api/models/ns/repo", "update"),
        ("DELETE", "/api/models/ns/repo", "delete"),
        ("DELETE", "/api/models/ns/repo/files/ns:repo:main:config.json", "delete"),
    ],
)
def test_repository_authorization_denial_happens_before_handlers(monkeypatch, method, path, expected_action):
    """Deny repository access after service access and before route/database work."""
    import gen3_ai_model_repo.auth as auth

    calls = []

    async def fake_authorize_request(*, authz_resources, authz_service_name, authz_access_method, request):
        del authz_service_name, request
        calls.append((authz_resources, authz_access_method))
        if authz_resources == ["/ai_model_repo/ns/repo"]:
            from fastapi import HTTPException

            raise HTTPException(status_code=403)

    monkeypatch.setattr(auth, "authorize_request", fake_authorize_request)
    app = FastAPI()
    app.include_router(ai_models_files_router)
    app.include_router(ai_models_repositories_router)
    app.include_router(ai_models_uploads_router)
    client = TestClient(app)
    response = client.request(method, path, json={"description": "must not run"})

    assert response.status_code == 403
    assert calls == [
        ([auth.config.AUTHZ_SERVICE_RESOURCE], "access"),
        (["/ai_model_repo/ns/repo"], expected_action),
    ]


def test_upload_rejects_too_many_files(monkeypatch):
    """Reject multipart uploads that exceed the configured file limit."""
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
    """Reject uploads whose declared content exceeds the configured limit."""
    import gen3_ai_model_repo.routes.ai_models_uploads as upload_routes

    monkeypatch.setattr(upload_routes.config, "MAX_UPLOAD_BYTES", 1)
    client = _build_test_client()

    response = client.post(
        "/api/models/ns/repo/upload",
        files=[("files", ("model.bin", b"model", "application/octet-stream"))],
    )

    assert response.status_code == 413


def test_update_repository_metadata(monkeypatch):
    """Update repository description and tags through the API."""
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
