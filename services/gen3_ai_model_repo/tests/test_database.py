"""Database operation tests using lightweight async connection fakes."""

import pytest

from gen3_ai_model_repo.database import file_tracking, repo_metadata, revisions


class FakeStatement:
    """Record prepared-statement calls made by database helpers."""

    def __init__(self, conn, query):
        """Initialize a statement bound to a fake connection."""
        self.conn = conn
        self.query = query

    async def execute(self, *args):
        """Record an execute call."""
        self.conn.executed.append((self.query, args))
        return "DELETE 1"

    async def fetchrow(self, *args):
        """Record a fetchrow call and return no row."""
        self.conn.executed.append((self.query, args))
        return None

    async def fetch(self, *args):
        """Record a fetch call and return no rows."""
        self.conn.executed.append((self.query, args))
        return []

    async def fetchval(self, *args):
        """Record a fetchval call with delete behavior."""
        self.conn.executed.append((self.query, args))
        if "DELETE" in self.query and "RETURNING" in self.query:
            return 1
        return None


class FakeConn:
    """Minimal async connection fake for repository database helpers."""

    def __init__(self):
        """Initialize empty call and row collections."""
        self.executed = []
        self.rows = []

    async def prepare(self, query):
        """Return a prepared statement fake."""
        return FakeStatement(self, query)

    async def execute(self, query, *args):
        """Record a direct execute call."""
        self.executed.append((query, args))
        return "DELETE 1"

    async def fetchrow(self, query, *args):
        """Record a fetchrow call and return no row."""
        self.executed.append((query, args))
        return None

    async def fetch(self, query, *args):
        """Record a fetch call and return no rows."""
        self.executed.append((query, args))
        return []

    async def fetchval(self, query, *args):
        """Record a fetchval call and return no value."""
        self.executed.append((query, args))
        return None


class FakeAcquire:
    """Async context manager for acquiring a fake connection."""

    def __init__(self, conn):
        """Initialize the acquired connection."""
        self.conn = conn

    async def __aenter__(self):
        """Return the fake connection."""
        return self.conn

    async def __aexit__(self, exc_type, exc, tb):
        """Leave the acquisition context without suppressing errors."""
        return False


class FakePool:
    """Minimal connection-pool fake."""

    def __init__(self, conn):
        """Initialize the pool with a connection."""
        self.conn = conn

    def acquire(self):
        """Return an async acquisition context manager."""
        return FakeAcquire(self.conn)


@pytest.mark.asyncio
async def test_delete_model_metadata(monkeypatch):
    """
    Verify model metadata delete returns True on successful delete.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture used to patch DB pool resolver.
    """
    conn = FakeConn()

    async def fake_get_db_pool():
        return FakePool(conn)

    monkeypatch.setattr(repo_metadata, "get_db_pool", fake_get_db_pool)
    assert await repo_metadata.delete_model_metadata("ns", "repo") is True


@pytest.mark.asyncio
async def test_track_file_false_when_missing_repo(monkeypatch):
    """
    Verify file tracking returns False when repository lookup fails.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture used to patch DB pool resolver.
    """

    class MissingRepoConn(FakeConn):
        async def fetchrow(self, query, *args):
            return None

    async def fake_get_db_pool():
        return FakePool(MissingRepoConn())

    monkeypatch.setattr(file_tracking, "get_db_pool", fake_get_db_pool)
    assert await file_tracking.track_file("ns", "repo", "main", "a.txt", 1, "sha") is False


@pytest.mark.asyncio
async def test_get_or_create_revision_none_when_missing_repo(monkeypatch):
    """
    Verify revision upsert returns None when repository does not exist.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture used to patch DB pool resolver.
    """

    async def fake_get_db_pool():
        return FakePool(FakeConn())

    monkeypatch.setattr(revisions, "get_db_pool", fake_get_db_pool)
    assert await revisions.get_or_create_revision("ns", "repo") is None


@pytest.mark.asyncio
async def test_list_models_uses_bound_array_scope_before_pagination(monkeypatch):
    """Authorization scope is bound once and applied before LIMIT/OFFSET."""
    conn = FakeConn()

    async def fake_get_db_pool():
        return FakePool(conn)

    monkeypatch.setattr(repo_metadata, "get_db_pool", fake_get_db_pool)
    scope = ["/ai_model_repo/ns/repo", "/ai_model_repo/other"] * 600
    await repo_metadata.list_models(permitted_resource_paths=scope, limit=25, offset=100)

    query, args = conn.executed[-1]
    assert "ANY($1::text[])" in query
    assert args[0] == scope
    assert query.index("WHERE") < query.index("ORDER BY") < query.index("LIMIT")
    assert args[-2:] == (25, 100)


@pytest.mark.asyncio
async def test_list_models_empty_scope_returns_before_database_query(monkeypatch):
    """No grants must not accidentally expose repositories or query all rows."""
    called = False

    async def fake_get_db_pool():
        nonlocal called
        called = True
        raise AssertionError("empty authorization scope must short-circuit")

    monkeypatch.setattr(repo_metadata, "get_db_pool", fake_get_db_pool)
    assert await repo_metadata.list_models(permitted_resource_paths=[]) == []
    assert called is False
