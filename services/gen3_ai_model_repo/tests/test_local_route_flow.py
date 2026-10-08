"""Black-box local-storage route regression tests."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from gen3_ai_model_repo.auth import verify_authorization
from gen3_ai_model_repo.routes.ai_models_files import ai_models_files_router
from gen3_ai_model_repo.routes.ai_models_repositories import ai_models_repositories_router
from gen3_ai_model_repo.routes.ai_models_uploads import ai_models_uploads_router
from gen3_ai_model_repo.storage.local import LocalStorageProvider


class _Statement:
    def __init__(self, inserted):
        self.inserted = inserted

    async def fetchval(self, *args):
        return None

    async def fetch(self, *args):
        if len(args) >= 6 and isinstance(args[5], str):
            self.inserted.append(args[5])


class _Transaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class _Conn:
    def __init__(self, inserted):
        self.inserted = inserted

    def transaction(self):
        return _Transaction()

    async def prepare(self, query):
        return _Statement(self.inserted)


class _Acquire:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, *args):
        return False


class _Pool:
    def __init__(self, conn):
        self.conn = conn

    def acquire(self):
        return _Acquire(self.conn)


def test_local_upload_resolve_head_and_delete_filesystem(monkeypatch, tmp_path: Path):
    """Exercise upload, resolve, HEAD, and all deletion routes with local files."""
    import gen3_ai_model_repo.routes.ai_models_files as file_routes
    import gen3_ai_model_repo.routes.ai_models_repositories as repo_routes
    import gen3_ai_model_repo.routes.ai_models_uploads as upload_routes

    provider = LocalStorageProvider(str(tmp_path))
    inserted: list[str] = []
    pool = _Pool(_Conn(inserted))
    record = {}
    repository_exists = False

    async def no_repo(*args):
        del args
        return repository_exists

    async def get_pool():
        return pool

    async def file_record(*args, **kwargs):
        del args
        return record or None

    async def fake_delete_file(*args):
        return True

    async def fake_storage_key_referenced(*args):
        return False

    async def fake_revision_keys(*args):
        return list(inserted)

    async def fake_repo_keys(*args):
        return list(inserted)

    async def fake_delete_revision(*args):
        return True

    async def fake_delete_files(*args):
        return 1

    async def fake_delete_repo(*args):
        return True

    monkeypatch.setattr(upload_routes, "db_model_exists", no_repo)
    monkeypatch.setattr(upload_routes, "get_db_pool", get_pool)
    monkeypatch.setattr(upload_routes, "get_storage_provider", lambda: provider)
    monkeypatch.setattr(file_routes, "get_storage_provider", lambda: provider)
    monkeypatch.setattr(file_routes, "get_file_record", file_record)
    monkeypatch.setattr(file_routes, "delete_file", fake_delete_file)
    monkeypatch.setattr(file_routes, "storage_key_is_referenced", fake_storage_key_referenced)
    monkeypatch.setattr(file_routes, "get_storage_keys_for_revision", fake_revision_keys)
    monkeypatch.setattr(file_routes, "delete_files_for_revision", fake_delete_files)
    monkeypatch.setattr(file_routes, "delete_revision", fake_delete_revision)
    monkeypatch.setattr(repo_routes, "db_model_exists", lambda *args: no_repo(*args))
    monkeypatch.setattr(repo_routes, "get_storage_provider", lambda: provider)
    monkeypatch.setattr(repo_routes, "get_storage_keys_for_repository", fake_repo_keys)
    monkeypatch.setattr(repo_routes, "storage_key_is_referenced", fake_storage_key_referenced)
    monkeypatch.setattr(repo_routes, "delete_model_metadata", fake_delete_repo)

    async def auth_override():
        return None

    app = FastAPI()
    app.include_router(ai_models_uploads_router)
    app.include_router(ai_models_files_router)
    app.include_router(ai_models_repositories_router)
    app.dependency_overrides[verify_authorization] = auth_override

    with TestClient(app) as client:
        response = client.post(
            "/api/models/ns/repo/upload",
            data={"revision_name": "main"},
            files={"files": ("config.json", b'{"ok":true}', "application/json")},
        )
        assert response.status_code == 200
        repository_exists = True
        key = inserted[0]
        record.update({"object_key": key, "size": 11, "sha": "commit", "etag": "etag", "path": "config.json"})
        stored = tmp_path / key
        assert stored.read_bytes() == b'{"ok":true}'

        resolved = client.get("/api/models/ns/repo/resolve/main/config.json")
        assert resolved.status_code == 200
        assert resolved.content == b'{"ok":true}'
        assert resolved.headers["x-repo-commit"] == "commit"
        assert resolved.headers["x-linked-etag"] == "etag"

        head = client.head("/api/models/ns/repo/resolve/main/config.json")
        assert head.status_code == 200
        assert head.headers["x-linked-size"] == "11"

        untracked = tmp_path / "ns/repo/direct-upload.bin"
        untracked.parent.mkdir(parents=True, exist_ok=True)
        untracked.write_bytes(b"untracked")
        assert client.delete("/api/models/ns/repo/files/ns:repo:main:config.json").status_code == 200
        assert not stored.exists()
        assert client.delete("/api/models/ns/repo/revisions/main").status_code == 200
        assert client.delete("/api/models/ns/repo").status_code == 200
        assert not untracked.exists()


def test_missing_local_storage_object_is_not_reported_as_available(monkeypatch, tmp_path: Path):
    """A stale DB row returns 404 for both GET and HEAD."""
    import gen3_ai_model_repo.routes.ai_models_files as file_routes

    provider = LocalStorageProvider(str(tmp_path))
    record = {"object_key": "ns/repo/main/missing.bin", "size": 1, "sha": "sha", "etag": "etag"}

    async def file_record(*args, **kwargs):
        return record

    async def auth_override():
        return None

    monkeypatch.setattr(file_routes, "get_file_record", file_record)
    monkeypatch.setattr(file_routes, "get_storage_provider", lambda: provider)
    app = FastAPI()
    app.include_router(ai_models_files_router)
    app.dependency_overrides[verify_authorization] = auth_override
    with TestClient(app) as client:
        assert client.get("/api/models/ns/repo/resolve/main/missing.bin").status_code == 404
        assert client.head("/api/models/ns/repo/resolve/main/missing.bin").status_code == 404
