"""
Tests for db/backfill_sha256_content_hashes.py.

The backfill's whole job is to make pre-existing rows agree with what the app now writes, so
these tests seed rows the way the old code did (md5-era hashes, NULL v2 columns) and check
that the hashes it computes match the app's, that it is idempotent, and that it reports
rather than swallows the rows it cannot hash.
"""

import asyncio
import importlib.util
from pathlib import Path

import asyncpg
import pytest
from pgvector.asyncpg import register_vector

from gen3_embeddings.database import hashing
from gen3_embeddings.models.schemas import VectorType

DOCS_AUTHZ = "/vectorstore/collections/docs"


def _load_backfill_module():
    """Import the backfill script, which lives in db/ rather than in the package."""
    path = Path(__file__).resolve().parents[1] / "db" / "backfill_sha256_content_hashes.py"
    spec = importlib.util.spec_from_file_location("backfill_sha256_content_hashes", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


backfill = _load_backfill_module()


async def _seed_legacy_rows(admin_dsn: str, table: str, vector_type: str, vectors: list[str]) -> None:
    """
    Insert rows the way the pre-sha256 code did: md5 hashes in v1, nothing in v2.

    Seeded over the admin DSN because these tables FORCE row level security and the rows need
    to exist regardless of any authz setting.
    """
    conn = await asyncpg.connect(admin_dsn)
    try:
        await conn.execute("TRUNCATE TABLE embeddings_vector, embeddings_halfvec, collections CASCADE")
        collection_id = await conn.fetchval(
            "INSERT INTO collections (collection_name, dimensions, vector_type) VALUES ('docs', 3, $1) RETURNING id",
            vector_type,
        )
        for vector in vectors:
            await conn.execute(
                f"""
                INSERT INTO {table} (collection_id, embedding, metadata, authz, embedding_hash, metadata_hash)
                VALUES ($1, $2::text::{vector_type}, '{{}}'::jsonb, $3, md5($2)::uuid, md5('{{}}')::uuid)
                """,
                collection_id,
                vector,
                DOCS_AUTHZ,
            )
    finally:
        await conn.close()


async def _run_backfill(admin_dsn: str, table: str, vector_type: VectorType, dry_run: bool = False) -> dict:
    """Run the backfill against one table and return its result summary."""
    conn = await asyncpg.connect(admin_dsn)
    await register_vector(conn)
    try:
        return await backfill.backfill_table(conn, table, vector_type, dry_run)
    finally:
        await conn.close()


async def _fetch_hashes(admin_dsn: str, table: str) -> list[asyncpg.Record]:
    """Read back the hash columns, ordered so assertions can index them."""
    conn = await asyncpg.connect(admin_dsn)
    await register_vector(conn)
    try:
        return await conn.fetch(
            f"SELECT embedding, embedding_hash, metadata_hash, embedding_hash_v2, metadata_hash_v2 "
            f"FROM {table} ORDER BY embedding_id"
        )
    finally:
        await conn.close()


@pytest.mark.parametrize(
    ("table", "vector_type"),
    [("embeddings_vector", VectorType.vector), ("embeddings_halfvec", VectorType.halfvec)],
)
def test_backfill_writes_the_hashes_the_app_would_write(test_database, table, vector_type):
    """
    A backfilled row ends up with the hash a fresh write of the same content would produce.

    That agreement is the point of the whole exercise: without it the new unique index cannot
    see an old row and a new row as the same embedding.
    """
    admin_dsn = test_database["admin_dsn"]
    vectors = ["[1,2,3]", "[4,5,6]"]
    asyncio.run(_seed_legacy_rows(admin_dsn, table, vector_type.value, vectors))

    result = asyncio.run(_run_backfill(admin_dsn, table, vector_type))

    assert result["updated"] == 2
    assert result["collisions"] == []

    rows = asyncio.run(_fetch_hashes(admin_dsn, table))
    for row in rows:
        expected = hashing.hash_vector(row["embedding"].to_list(), vector_type, 3)
        assert row["embedding_hash_v2"] == expected
        assert row["metadata_hash_v2"] == hashing.hash_metadata({})
        # the legacy columns are left exactly as they were
        assert row["embedding_hash"] != row["embedding_hash_v2"]


def test_backfill_is_idempotent(test_database):
    """A second run finds nothing to do, so an interrupted run can just be re-run."""
    admin_dsn = test_database["admin_dsn"]
    asyncio.run(_seed_legacy_rows(admin_dsn, "embeddings_vector", "vector", ["[1,2,3]"]))

    first = asyncio.run(_run_backfill(admin_dsn, "embeddings_vector", VectorType.vector))
    second = asyncio.run(_run_backfill(admin_dsn, "embeddings_vector", VectorType.vector))

    assert first["updated"] == 1
    assert second["scanned"] == 0
    assert second["updated"] == 0


def test_dry_run_writes_nothing(test_database):
    """--dry-run reports what it would do and leaves the columns NULL."""
    admin_dsn = test_database["admin_dsn"]
    asyncio.run(_seed_legacy_rows(admin_dsn, "embeddings_vector", "vector", ["[1,2,3]"]))

    result = asyncio.run(_run_backfill(admin_dsn, "embeddings_vector", VectorType.vector, dry_run=True))

    assert result["updated"] == 1
    rows = asyncio.run(_fetch_hashes(admin_dsn, "embeddings_vector"))
    assert rows[0]["embedding_hash_v2"] is None


def test_pre_existing_duplicates_are_reported_not_hidden(test_database):
    """
    Rows the old hashes let in as distinct, but which are actually one embedding, are reported.

    On a halfvec collection 1.0 and 1.0001 store identically, so md5 over the JSON text wrote
    two rows for what is one embedding. Only one of them can hold the shared hash; the other
    is left NULL and named in the summary so a human decides which to keep. Nothing is deleted.
    """
    admin_dsn = test_database["admin_dsn"]
    asyncio.run(_seed_legacy_rows(admin_dsn, "embeddings_halfvec", "halfvec", ["[1.0,2,3]", "[1.0001,2,3]"]))

    result = asyncio.run(_run_backfill(admin_dsn, "embeddings_halfvec", VectorType.halfvec))

    assert result["updated"] == 1
    assert len(result["collisions"]) == 1

    rows = asyncio.run(_fetch_hashes(admin_dsn, "embeddings_halfvec"))
    hashed = [row for row in rows if row["embedding_hash_v2"] is not None]
    assert len(hashed) == 1
