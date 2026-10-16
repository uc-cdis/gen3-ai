"""
Tests for db/seed.py.

The seed exists so local data cannot drift from what the service writes, so these run it as the
app role and check the rows carry the authz, hashes and names the API would have given them.
"""

import importlib.util
from pathlib import Path

import asyncpg
import pytest


def _load_seed_module():
    """Import the seed script, which lives in db/ rather than in the package."""
    path = Path(__file__).resolve().parents[1] / "db" / "seed.py"
    spec = importlib.util.spec_from_file_location("seed", path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


seed = _load_seed_module()


async def _rows_per_collection(admin_dsn: str) -> dict[str, tuple[int, set[str], int]]:
    """Per collection: (embedding count, distinct authz values, rows missing a v2 hash), read as admin."""
    conn = await asyncpg.connect(admin_dsn)
    try:
        rows = await conn.fetch(
            """
            SELECT c.collection_name,
                   count(e.embedding_id) AS n,
                   coalesce(array_agg(DISTINCT e.authz) FILTER (WHERE e.authz IS NOT NULL), '{}') AS authz,
                   count(*) FILTER (WHERE e.embedding_id IS NOT NULL AND e.embedding_hash_v2 IS NULL) AS unhashed
            FROM collections c
            LEFT JOIN (
                SELECT collection_id, embedding_id, authz, embedding_hash_v2 FROM embeddings_vector
                UNION ALL
                SELECT collection_id, embedding_id, authz, embedding_hash_v2 FROM embeddings_halfvec
            ) e ON e.collection_id = c.id
            GROUP BY c.collection_name
            """
        )
    finally:
        await conn.close()
    return {row["collection_name"]: (row["n"], set(row["authz"]), row["unhashed"]) for row in rows}


@pytest.mark.asyncio
async def test_seed_writes_every_collection_as_the_app_role(test_database, reset_db):
    """Every collection lands with its rows, under its own authz path, hashed the way the service hashes."""
    assert await seed.main(test_database["app_dsn"], reset=False) == 0

    by_name = await _rows_per_collection(test_database["admin_dsn"])
    assert by_name == {s.name: (s.rows, {f"/vectorstore/collections/{s.name}"}, 0) for s in seed.SEED_COLLECTIONS}


@pytest.mark.asyncio
async def test_seed_rerun_skips_and_reset_reloads(test_database, reset_db, capsys):
    """A second run writes nothing; --reset replaces each collection rather than adding to it."""
    assert await seed.main(test_database["app_dsn"], reset=False) == 0
    capsys.readouterr()

    assert await seed.main(test_database["app_dsn"], reset=False) == 0
    assert "Done, 0 embeddings written." in capsys.readouterr().out

    assert await seed.main(test_database["app_dsn"], reset=True) == 0
    total = sum(s.rows for s in seed.SEED_COLLECTIONS)
    assert f"Done, {total} embeddings written." in capsys.readouterr().out

    by_name = await _rows_per_collection(test_database["admin_dsn"])
    assert {name: n for name, (n, _, _) in by_name.items()} == {s.name: s.rows for s in seed.SEED_COLLECTIONS}


@pytest.mark.asyncio
async def test_seed_refuses_a_role_that_bypasses_rls(test_database, reset_db):
    """Seeding as the superuser would pass whatever the policies say, so it is refused before writing."""
    assert await seed.main(test_database["admin_dsn"], reset=False) == 1
    assert await _rows_per_collection(test_database["admin_dsn"]) == {}
