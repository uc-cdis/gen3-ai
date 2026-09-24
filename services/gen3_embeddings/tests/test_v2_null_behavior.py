"""
What the API does against rows whose v2 content hashes have not been backfilled yet.

The perf database has 30M rows written before 20260826120000 added the sha256 columns, so
`embedding_hash_v2`/`metadata_hash_v2` are NULL on all of them and the backfill is expensive
enough to be worth skipping. These tests pin down exactly what that costs.

A pre-backfill row is simulated rather than assumed: its v2 columns are NULLed and its legacy
columns are set to an md5-era value, which is precisely the state the backfill exists to fix.
"""

import asyncio
import uuid

import asyncpg
import pytest


def _create_collection(client, name, dimensions=3):
    """Create a collection and fail loudly if it did not take."""
    response = client.post(
        "/vectorstore/collections",
        json={"collection_name": name, "description": "test", "dimensions": dimensions, "vector_type": "vector"},
    )
    assert response.status_code == 200, response.text


def _make_pre_backfill(admin_dsn, embedding_id):
    """Rewind one row to the state rows were in before the sha256 migration."""

    async def _run():
        conn = await asyncpg.connect(admin_dsn)
        try:
            await conn.execute(
                """
                UPDATE embeddings_vector
                SET embedding_hash_v2 = NULL,
                    metadata_hash_v2 = NULL,
                    embedding_hash = $2::uuid,
                    metadata_hash = $3::uuid
                WHERE embedding_id = $1::uuid
                """,
                embedding_id,
                uuid.uuid4(),  # stands in for the old md5-over-JSON-text hash
                uuid.uuid4(),
            )
        finally:
            await conn.close()

    asyncio.run(_run())


def _post(client, vector, metadata=None):
    """POST one embedding into the `docs` collection."""
    return client.post(
        "/vectorstore/collections/docs/embeddings",
        json={"embeddings": [{"embedding": vector, "metadata": metadata or {}}]},
    )


@pytest.fixture
def seeded(client, allow_authz, test_database):
    """A `docs` collection holding one embedding, fully hashed as the app writes it today."""
    allow_authz("docs")
    _create_collection(client, "docs")
    response = _post(client, [0.1, 0.2, 0.3], {"a": 1})
    assert response.status_code == 200, response.text
    return response.json()["embeddings"][0]["embedding_id"]


def test_a_fully_hashed_row_still_rejects_a_duplicate(client, seeded):
    """The baseline: with v2 populated, re-posting the same content is a conflict."""
    assert _post(client, [0.1, 0.2, 0.3], {"a": 1}).status_code == 409


def test_a_pre_backfill_row_does_not_reject_a_duplicate(client, seeded, test_database):
    """
    This is what skipping the backfill costs: the same content lands twice.

    The v2 unique index is NULLS DISTINCT, so a NULL v2 row collides with nothing, and the
    legacy index cannot catch it either -- the app now writes the sha256 value into the legacy
    columns, which will never equal the md5 an old row holds there.
    """
    _make_pre_backfill(test_database["admin_dsn"], seeded)

    response = _post(client, [0.1, 0.2, 0.3], {"a": 1})

    assert response.status_code == 200, response.text
    assert response.json()["embeddings"][0]["embedding_id"] != seeded


def test_a_pre_backfill_row_is_not_matched_by_an_upsert(client, seeded, test_database):
    """
    PUT conflicts on the v2 columns, so a NULL-v2 row is invisible to it.

    Worse than the POST case in one respect: the caller asked to update whatever already held
    this content and instead silently gained a second row holding it.
    """
    _make_pre_backfill(test_database["admin_dsn"], seeded)

    response = client.put(
        "/vectorstore/collections/docs/embeddings",
        json={"embeddings": [{"embedding": [0.1, 0.2, 0.3], "metadata": {"a": 1}}]},
    )

    assert response.status_code == 200, response.text
    assert response.json()["embeddings"][0]["embedding_id"] != seeded


def test_search_is_unaffected_by_null_v2(client, seeded, test_database):
    """Reads never touch the hash columns, so search behaves identically either way."""
    _make_pre_backfill(test_database["admin_dsn"], seeded)

    response = client.post(
        "/vectorstore/collections/docs/search",
        json={"input": [0.1, 0.2, 0.3], "top_k": 5, "distance_metric": "cosine_similarity"},
    )

    assert response.status_code == 200, response.text
    hits = response.json()["embeddings"]
    assert len(hits) == 1
    assert hits[0]["id"] == seeded
