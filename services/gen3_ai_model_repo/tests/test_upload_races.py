"""Regression tests for upload staging and persisted storage keys."""

from io import BytesIO

import pytest
from fastapi import HTTPException, UploadFile
from starlette.requests import Request


def _request() -> Request:
    """Build a typed request object for direct route invocation."""
    return Request({"type": "http", "method": "POST", "path": "/", "headers": []})


class _Provider:
    def __init__(self):
        self.uploaded = {}
        self.deleted = []

    async def upload_stream(self, stream, key):
        self.uploaded[key] = stream.read()

    async def get_file_metadata(self, key):
        return {"size": len(self.uploaded[key]), "etag": "etag"}

    async def delete_file(self, key):
        self.deleted.append(key)
        self.uploaded.pop(key, None)


class _Transaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _Acquire:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _Pool:
    def __init__(self, conn):
        self.conn = conn

    def acquire(self):
        return _Acquire(self.conn)


@pytest.mark.asyncio
async def test_competing_upload_cleans_only_attempt_objects(monkeypatch):
    """A transaction conflict must not remove an earlier repository's object."""
    import gen3_ai_model_repo.routes.ai_models_uploads as uploads

    provider = _Provider()
    provider.uploaded["ns/repo/main/original/config.json"] = b"original"

    async def conflict(*args):
        raise HTTPException(status_code=409, detail="Repository already exists")

    async def _false():
        return False

    monkeypatch.setattr(uploads, "db_model_exists", lambda *_: _false())
    monkeypatch.setattr(uploads, "_create_model_and_initial_revision", conflict)
    monkeypatch.setattr(uploads, "get_storage_provider", lambda: provider)

    class _Conn:
        def transaction(self):
            return _Transaction()

    def _pool():
        return _Pool(_Conn())

    async def get_pool():
        return _pool()

    monkeypatch.setattr(uploads, "get_db_pool", get_pool)

    upload = UploadFile(file=BytesIO(b"new"), filename="config.json")
    with pytest.raises(HTTPException) as error:
        await uploads.upload_model(
            request=_request(),
            namespace="ns",
            repo="repo",
            revision_name="main",
            files=[upload],
        )

    assert error.value.status_code == 409
    assert provider.uploaded["ns/repo/main/original/config.json"] == b"original"
    assert len(provider.deleted) == 1
    assert provider.deleted[0].startswith("ns/repo/main/")
    assert "/config.json" in provider.deleted[0]


@pytest.mark.asyncio
async def test_successful_upload_persists_attempt_specific_key(monkeypatch):
    """The DB, download, and cleanup path must use the actual staged key."""
    import gen3_ai_model_repo.routes.ai_models_uploads as uploads

    provider = _Provider()
    inserted = []

    async def no_repo(*args):
        return False

    async def create_model(*args):
        return 7, 8

    class _Statement:
        async def fetch(self, *args):
            inserted.append(args)

    class _Conn:
        def transaction(self):
            return _Transaction()

        async def prepare(self, query):
            return _Statement()

    monkeypatch.setattr(uploads, "db_model_exists", no_repo)
    monkeypatch.setattr(uploads, "_create_model_and_initial_revision", create_model)
    monkeypatch.setattr(uploads, "get_storage_provider", lambda: provider)

    async def get_pool():
        return _Pool(_Conn())

    monkeypatch.setattr(uploads, "get_db_pool", get_pool)

    upload = UploadFile(file=BytesIO(b"content"), filename="config.json")
    result = await uploads.upload_model(
        request=_request(),
        namespace="ns",
        repo="repo",
        revision_name="main",
        files=[upload],
    )

    assert result.status == "uploaded"
    stored_key = next(iter(provider.uploaded))
    assert stored_key.startswith("ns/repo/main/")
    assert stored_key != "ns/repo/main/config.json"
    assert inserted[0][5] == stored_key
