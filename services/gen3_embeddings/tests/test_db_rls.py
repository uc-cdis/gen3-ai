"""
Behavioral tests that row-level security is enforced by Postgres itself.

test_db_bootstrap.py asserts RLS is *configured* (enabled, policies present). These tests
assert it actually *works*: they connect as the unprivileged app user and check which rows
are visible for a given `app.allowed_authz`, independent of any application-layer authz.

Rows are seeded over the admin DSN because that role is a superuser and so bypasses RLS,
which keeps setup separate from the behavior under test.
"""

import asyncpg
import pytest

DOCS_AUTHZ = "/vectorstore/collections/docs"
IMAGES_AUTHZ = "/vectorstore/collections/images"

EMBEDDING_TABLES = ("embeddings_vector", "embeddings_halfvec")


async def _seed_two_authz_paths(admin_dsn: str) -> None:
    """Put one row per authz path into both embedding tables, as the RLS-exempt superuser."""
    conn = await asyncpg.connect(admin_dsn)
    try:
        await conn.execute("TRUNCATE TABLE embeddings_vector, embeddings_halfvec, collections CASCADE")
        for name, vector_type in (("docs", "vector"), ("images", "halfvec")):
            await conn.execute(
                "INSERT INTO collections (collection_name, dimensions, vector_type) VALUES ($1, 3, $2)",
                name,
                vector_type,
            )
        # every table gets a row for BOTH authz paths, so each test can prove that one path
        # is visible and the other is not
        for table in EMBEDDING_TABLES:
            for authz in (DOCS_AUTHZ, IMAGES_AUTHZ):
                await conn.execute(
                    f"""
                    INSERT INTO {table} (collection_id, embedding, embedding_hash, authz)
                    VALUES ((SELECT min(id) FROM collections), '[1,0,0]', gen_random_uuid(), $1)
                    """,
                    authz,
                )
    finally:
        await conn.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("table", EMBEDDING_TABLES)
async def test_rls_scopes_visible_rows_to_allowed_authz(test_database, table):
    """With one authz path set, only that path's rows are visible; the other is filtered out."""
    await _seed_two_authz_paths(test_database["admin_dsn"])

    conn = await asyncpg.connect(test_database["app_dsn"])
    try:
        await conn.execute("SELECT set_config('app.allowed_authz', $1::text[]::text, false)", [DOCS_AUTHZ])

        visible = await conn.fetch(f"SELECT authz FROM {table}")
        assert [r["authz"] for r in visible] == [DOCS_AUTHZ]

        # widening the setting reveals the second row, confirming it was there all along
        await conn.execute(
            "SELECT set_config('app.allowed_authz', $1::text[]::text, false)", [DOCS_AUTHZ, IMAGES_AUTHZ]
        )
        widened = await conn.fetch(f"SELECT authz FROM {table} ORDER BY authz")
        assert [r["authz"] for r in widened] == [DOCS_AUTHZ, IMAGES_AUTHZ]
    finally:
        await conn.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("table", EMBEDDING_TABLES)
async def test_rls_denies_all_rows_when_authz_never_set(test_database, table):
    """A connection that never sets app.allowed_authz sees nothing, rather than everything."""
    await _seed_two_authz_paths(test_database["admin_dsn"])

    conn = await asyncpg.connect(test_database["app_dsn"])
    try:
        assert await conn.fetchval(f"SELECT count(*) FROM {table}") == 0
    finally:
        await conn.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("table", EMBEDDING_TABLES)
async def test_rls_fails_closed_on_empty_authz_setting(test_database, table):
    """
    An empty app.allowed_authz denies access instead of raising.

    Regression test: the original policy cast the setting straight to text[], and ''::text[]
    is a `malformed array literal` error rather than an empty array.
    """
    await _seed_two_authz_paths(test_database["admin_dsn"])

    conn = await asyncpg.connect(test_database["app_dsn"])
    try:
        await conn.execute("SELECT set_config('app.allowed_authz', '', false)")
        assert await conn.fetchval(f"SELECT count(*) FROM {table}") == 0
    finally:
        await conn.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("table", EMBEDDING_TABLES)
async def test_rls_fails_closed_after_transaction_local_setting_reverts(test_database, table):
    """
    Reusing a connection after an RLS-scoped transaction denies access instead of raising.

    The DAL sets app.allowed_authz with is_local=true, so at the end of that transaction the
    value reverts to the empty string, not to NULL. Pooled connections are reused, so a later
    query outside such a transaction must fail closed rather than error.
    """
    await _seed_two_authz_paths(test_database["admin_dsn"])

    conn = await asyncpg.connect(test_database["app_dsn"])
    try:
        async with conn.transaction():
            await conn.execute("SELECT set_config('app.allowed_authz', $1::text[]::text, true)", [DOCS_AUTHZ])
            assert await conn.fetchval(f"SELECT count(*) FROM {table}") == 1

        # same connection, now outside the transaction: the setting has reverted to ''
        assert await conn.fetchval(f"SELECT count(*) FROM {table}") == 0
    finally:
        await conn.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("table", EMBEDDING_TABLES)
async def test_rls_with_check_rejects_insert_outside_allowed_authz(test_database, table):
    """WITH CHECK blocks writing a row under an authz path the caller does not hold."""
    await _seed_two_authz_paths(test_database["admin_dsn"])

    conn = await asyncpg.connect(test_database["app_dsn"])
    try:
        await conn.execute("SELECT set_config('app.allowed_authz', $1::text[]::text, false)", [DOCS_AUTHZ])

        insert = f"""
            INSERT INTO {table} (collection_id, embedding, embedding_hash, authz)
            VALUES ((SELECT min(id) FROM collections), '[0,1,0]', gen_random_uuid(), $1)
        """

        with pytest.raises(asyncpg.exceptions.InsufficientPrivilegeError):
            await conn.execute(insert, IMAGES_AUTHZ)

        # the same insert under an allowed authz path succeeds
        await conn.execute(insert, DOCS_AUTHZ)
        assert await conn.fetchval(f"SELECT count(*) FROM {table}") == 2
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_rls_is_forced_so_table_owners_cannot_bypass_it(test_database):
    """
    Both embedding tables have RLS FORCEd.

    Without FORCE, whichever role owns these tables bypasses every policy silently. The
    service's startup check rejects SUPERUSER and BYPASSRLS but does not check ownership,
    so FORCE is what makes RLS hold if the app ever connects as the owner.
    """
    conn = await asyncpg.connect(test_database["app_dsn"])
    try:
        rows = await conn.fetch(
            """
            SELECT c.relname, c.relforcerowsecurity
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = 'public' AND c.relname = ANY($1::text[])
            """,
            list(EMBEDDING_TABLES),
        )
        forced = {r["relname"]: r["relforcerowsecurity"] for r in rows}
        assert forced == {"embeddings_vector": True, "embeddings_halfvec": True}
    finally:
        await conn.close()
