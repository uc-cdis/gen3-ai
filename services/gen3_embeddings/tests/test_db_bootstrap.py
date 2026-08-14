import pytest
import asyncpg


@pytest.mark.asyncio
async def test_app_user_can_connect_and_query(test_database):
    """The app user's DSN is valid and the database accepts connections."""
    conn = await asyncpg.connect(test_database["app_dsn"])
    try:
        value = await conn.fetchval("SELECT 1")
        assert value == 1
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_app_user_is_not_superuser_or_bypassrls(test_database):
    """The app user has no superuser or bypassrls privileges, enforcing row-level security."""
    conn = await asyncpg.connect(test_database["app_dsn"])
    try:
        row = await conn.fetchrow(
            """
            SELECT usesuper, usebypassrls, usename
            FROM pg_user
            WHERE usename = current_user
            """
        )
        assert row["usesuper"] is False
        assert row["usebypassrls"] is False
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_tables_exist(test_database):
    """Migrations create the collections, embeddings_vector, and embeddings_halfvec tables."""
    conn = await asyncpg.connect(test_database["app_dsn"])
    try:
        rows = await conn.fetch(
            """
            SELECT tablename
            FROM pg_tables
            WHERE schemaname = 'public'
            """
        )
        names = {r["tablename"] for r in rows}
        assert "collections" in names
        assert "embeddings_vector" in names
        assert "embeddings_halfvec" in names
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_rls_enabled_and_policies_exist(test_database):
    """Row-level security is enabled on both embedding tables and the expected authz policies are present."""
    conn = await asyncpg.connect(test_database["app_dsn"])
    try:
        rows = await conn.fetch(
            """
            SELECT tablename, rowsecurity
            FROM pg_tables
            JOIN pg_class ON pg_class.relname = pg_tables.tablename
            JOIN pg_namespace ns ON ns.oid = pg_class.relnamespace
            WHERE ns.nspname = 'public'
              AND tablename IN ('embeddings_vector', 'embeddings_halfvec')
            """
        )
        by_table = {r["tablename"]: r["rowsecurity"] for r in rows}
        assert by_table["embeddings_vector"] is True
        assert by_table["embeddings_halfvec"] is True

        policy_rows = await conn.fetch(
            """
            SELECT tablename, policyname
            FROM pg_policies
            WHERE schemaname = 'public'
              AND tablename IN ('embeddings_vector', 'embeddings_halfvec')
            """
        )
        policies = {(r["tablename"], r["policyname"]) for r in policy_rows}
        assert ("embeddings_vector", "authz_policy_vector") in policies
        assert ("embeddings_halfvec", "authz_policy_halfvec") in policies
    finally:
        await conn.close()
