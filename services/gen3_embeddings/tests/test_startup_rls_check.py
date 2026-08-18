"""
Tests for the startup guard that row-level security is actually in effect.

`check_db_connection`'s superuser/bypassrls checks only cover the ROLE side of RLS, so they
pass against a database where RLS has been switched off entirely. `check_rls_is_enabled` is
the fallback for that, since otherwise nothing but the migrations enforces it.
"""

import asyncpg
import pytest

from gen3_embeddings import config
from gen3_embeddings import main as main_module
from gen3_embeddings.database import db as db_module
from gen3_embeddings.main import RLS_PROTECTED_TABLES, check_rls_is_enabled


async def _set_rls(admin_dsn: str, table: str, *, enabled: bool, forced: bool) -> None:
    """Toggle RLS on a table over a connection that is allowed to alter it."""
    conn = await asyncpg.connect(admin_dsn)
    try:
        await conn.execute(f"ALTER TABLE {table} {'ENABLE' if enabled else 'DISABLE'} ROW LEVEL SECURITY")
        await conn.execute(f"ALTER TABLE {table} {'FORCE' if forced else 'NO FORCE'} ROW LEVEL SECURITY")
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_passes_on_a_migrated_database(test_database, monkeypatch):
    """The migrations leave RLS enabled and forced, so the check is satisfied."""
    monkeypatch.setattr(config, "DEBUG_SKIP_AUTH", False)

    conn = await asyncpg.connect(test_database["app_dsn"])
    try:
        await check_rls_is_enabled(conn)  # must not raise
    finally:
        await conn.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("table", RLS_PROTECTED_TABLES)
async def test_raises_when_rls_is_disabled(test_database, monkeypatch, table):
    """Disabling RLS makes every row visible to everyone, so startup must fail."""
    monkeypatch.setattr(config, "DEBUG_SKIP_AUTH", False)

    await _set_rls(test_database["admin_dsn"], table, enabled=False, forced=True)
    conn = await asyncpg.connect(test_database["app_dsn"])
    try:
        with pytest.raises(Exception, match="row-level security is DISABLED"):
            await check_rls_is_enabled(conn)
    finally:
        await conn.close()
        await _set_rls(test_database["admin_dsn"], table, enabled=True, forced=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("table", RLS_PROTECTED_TABLES)
async def test_raises_when_rls_is_not_forced(test_database, monkeypatch, table):
    """RLS that is enabled but not forced is bypassed by the table owner, so startup must fail."""
    monkeypatch.setattr(config, "DEBUG_SKIP_AUTH", False)

    await _set_rls(test_database["admin_dsn"], table, enabled=True, forced=False)
    conn = await asyncpg.connect(test_database["app_dsn"])
    try:
        with pytest.raises(Exception, match="row-level security is not FORCED"):
            await check_rls_is_enabled(conn)
    finally:
        await conn.close()
        await _set_rls(test_database["admin_dsn"], table, enabled=True, forced=True)


@pytest.mark.asyncio
async def test_reports_every_problem_at_once(test_database, monkeypatch):
    """Both tables being unprotected is reported together, not one failure at a time."""
    monkeypatch.setattr(config, "DEBUG_SKIP_AUTH", False)

    for table in RLS_PROTECTED_TABLES:
        await _set_rls(test_database["admin_dsn"], table, enabled=False, forced=False)

    conn = await asyncpg.connect(test_database["app_dsn"])
    try:
        with pytest.raises(Exception) as exc_info:
            await check_rls_is_enabled(conn)

        message = str(exc_info.value)
        for table in RLS_PROTECTED_TABLES:
            assert table in message
    finally:
        await conn.close()
        for table in RLS_PROTECTED_TABLES:
            await _set_rls(test_database["admin_dsn"], table, enabled=True, forced=True)


@pytest.mark.asyncio
async def test_check_db_connection_runs_the_rls_check(test_database, monkeypatch):
    """
    The RLS check is reached from the startup path, not just importable.

    Guards the wiring: without this, removing the call from check_db_connection would leave
    every test above passing while the running service no longer verified anything.
    """
    monkeypatch.setattr(config, "DEBUG_SKIP_AUTH", False)
    monkeypatch.setattr(config, "DB_CONNECTION_STRING", test_database["app_dsn"])

    called = False

    async def _record(conn):
        nonlocal called
        called = True

    monkeypatch.setattr(main_module, "check_rls_is_enabled", _record)

    monkeypatch.setattr(db_module, "_pool", None)
    try:
        await main_module.check_db_connection()
    finally:
        await db_module.close_pool()

    assert called, "check_db_connection did not invoke check_rls_is_enabled"


@pytest.mark.asyncio
async def test_downgrades_to_a_warning_under_debug_skip_auth(test_database, monkeypatch, caplog):
    """DEBUG_SKIP_AUTH keeps local development usable, matching the superuser check's behavior."""
    monkeypatch.setattr(config, "DEBUG_SKIP_AUTH", True)

    table = RLS_PROTECTED_TABLES[0]
    await _set_rls(test_database["admin_dsn"], table, enabled=False, forced=False)

    conn = await asyncpg.connect(test_database["app_dsn"])
    try:
        await check_rls_is_enabled(conn)  # must not raise
        assert "row-level security problem" in caplog.text
    finally:
        await conn.close()
        await _set_rls(test_database["admin_dsn"], table, enabled=True, forced=True)
