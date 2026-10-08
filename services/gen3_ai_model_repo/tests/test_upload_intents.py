"""Tests for durable direct-upload reservations."""

import pytest


class _Conn:
    def __init__(self, used=0):
        self.used = used
        self.args = []

    def transaction(self):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def fetchval(self, query, *args):
        self.args.append((query, args))
        return self.used

    async def execute(self, query, *args):
        self.args.append((query, args))


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


@pytest.mark.asyncio
async def test_direct_upload_quota_exhaustion(monkeypatch):
    """A reservation cannot exceed the durable byte quota."""
    from gen3_ai_model_repo import config
    from gen3_ai_model_repo.database import upload_intents

    conn = _Conn(used=config.DIRECT_UPLOAD_QUOTA_BYTES)

    async def pool():
        return _Pool(conn)

    monkeypatch.setattr(upload_intents, "get_db_pool", pool)
    with pytest.raises(ValueError, match="quota exhausted"):
        await upload_intents.reserve_upload("ns", "repo", "main", "x", "ns/repo/main/x")
    assert not any("INSERT INTO" in query for query, _ in conn.args)


@pytest.mark.asyncio
async def test_direct_upload_reservation_has_expiry_and_bound_completion(monkeypatch):
    """Reservations carry expiry and completion accepts only the capped ID list."""
    from gen3_ai_model_repo.database import upload_intents

    conn = _Conn()

    async def pool():
        return _Pool(conn)

    monkeypatch.setattr(upload_intents, "get_db_pool", pool)
    intent_id = await upload_intents.reserve_upload("ns", "repo", "main", "x", "ns/repo/main/x")
    insert_query, insert_args = conn.args[-1]
    assert "expires_at" in insert_query
    assert insert_args[0] == intent_id
    assert await upload_intents.get_upload_intents([intent_id] * 101, "ns", "repo", "main") == []
