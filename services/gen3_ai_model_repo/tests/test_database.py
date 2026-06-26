import pytest

from gen3_ai_model_repo.database import repo_metadata, revisions, file_tracking


class FakeConn:
    def __init__(self):
        self.executed = []
        self.rows = []

    async def execute(self, query, *args):
        self.executed.append((query, args))
        return "DELETE 1"

    async def fetchrow(self, query, *args):
        return None

    async def fetch(self, query, *args):
        return []

    async def fetchval(self, query, *args):
        return None


class FakeAcquire:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakePool:
    def __init__(self, conn):
        self.conn = conn

    def acquire(self):
        return FakeAcquire(self.conn)


@pytest.mark.anyio
async def test_delete_repository_metadata(monkeypatch):
    conn = FakeConn()

    async def fake_get_db_pool():
        return FakePool(conn)

    monkeypatch.setattr(repo_metadata, "get_db_pool", fake_get_db_pool)
    assert await repo_metadata.delete_repository_metadata("ns", "repo") is True


@pytest.mark.anyio
async def test_track_file_false_when_missing_repo(monkeypatch):
    class MissingRepoConn(FakeConn):
        async def fetchrow(self, query, *args):
            return None

    async def fake_get_db_pool():
        return FakePool(MissingRepoConn())

    monkeypatch.setattr(file_tracking, "get_db_pool", fake_get_db_pool)
    assert await file_tracking.track_file("ns", "repo", "main", "a.txt", 1, "sha") is False


@pytest.mark.anyio
async def test_get_or_create_revision_none_when_missing_repo(monkeypatch):
    async def fake_get_db_pool():
        return FakePool(FakeConn())

    monkeypatch.setattr(revisions, "get_db_pool", fake_get_db_pool)
    assert await revisions.get_or_create_revision("ns", "repo") is None
