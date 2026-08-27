"""
Tests for embedding deduplication now that content hashes are sha256 over stored bytes.

Two things are under test here. First, that the hashes actually reach the database and drive
the unique constraint (the app writes them; nothing recomputes them in SQL any more). Second,
that duplicates are recognized at STORAGE precision -- inputs that differ only in digits the
column cannot hold are the same row, which the previous md5-over-JSON-text hash could not see.
"""

import asyncio

import asyncpg
import pytest

from gen3_embeddings.database import hashing
from gen3_embeddings.models.schemas import VectorType


def _create_collection(client, name, dimensions=3, vector_type="vector"):
    """Create a collection and fail loudly if it did not take."""
    response = client.post(
        "/vectorstore/collections",
        json={
            "collection_name": name,
            "description": "test",
            "dimensions": dimensions,
            "vector_type": vector_type,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_created_rows_carry_the_apps_sha256_hashes(client, allow_authz, test_database):
    """
    The row's hash columns hold exactly what database/hashing.py computes.

    This is the contract the backfill and the unique index both depend on: a row written by
    the app and a row hashed offline have to agree.
    """
    allow_authz("docs")
    _create_collection(client, "docs")

    vector = [0.1, 0.2, 0.3]
    metadata = {"b": 1, "a": 2}
    response = client.post(
        "/vectorstore/collections/docs/embeddings",
        json={"embeddings": [{"embedding": vector, "metadata": metadata}]},
    )
    assert response.status_code == 200, response.text
    embedding_id = response.json()["embeddings"][0]["embedding_id"]

    async def _read():
        conn = await asyncpg.connect(test_database["admin_dsn"])
        try:
            return await conn.fetchrow(
                "SELECT embedding_hash, metadata_hash, embedding_hash_v2, metadata_hash_v2 "
                "FROM embeddings_vector WHERE embedding_id = $1::uuid",
                embedding_id,
            )
        finally:
            await conn.close()

    row = asyncio.run(_read())

    assert row["embedding_hash_v2"] == hashing.hash_vector(vector, VectorType.vector, 3)
    assert row["metadata_hash_v2"] == hashing.hash_metadata(metadata)
    # the legacy md5 columns are dual-written with the same value so their NOT NULL and unique
    # constraint hold until the contract migration drops them
    assert row["embedding_hash"] == row["embedding_hash_v2"]
    assert row["metadata_hash"] == row["metadata_hash_v2"]


def test_reposting_identical_content_conflicts(client, allow_authz):
    """Identical content in a second POST is a 409"""
    allow_authz("docs")
    _create_collection(client, "docs")

    body = {"embeddings": [{"embedding": [1.0, 2.0, 3.0], "metadata": {"a": 1}}]}
    assert client.post("/vectorstore/collections/docs/embeddings", json=body).status_code == 200

    conflict = client.post("/vectorstore/collections/docs/embeddings", json=body)
    assert conflict.status_code == 409, conflict.text


def test_metadata_key_order_does_not_create_a_second_row(client, allow_authz):
    """
    The same metadata written with different key order is the same row.

    The hash is over canonical JSON, so ordering cannot smuggle a duplicate past the
    constraint.
    """
    allow_authz("docs")
    _create_collection(client, "docs")

    first = client.post(
        "/vectorstore/collections/docs/embeddings",
        json={"embeddings": [{"embedding": [1.0, 2.0, 3.0], "metadata": {"a": 1, "b": 2}}]},
    )
    assert first.status_code == 200, first.text

    second = client.post(
        "/vectorstore/collections/docs/embeddings",
        json={"embeddings": [{"embedding": [1.0, 2.0, 3.0], "metadata": {"b": 2, "a": 1}}]},
    )
    assert second.status_code == 409, second.text


def test_halfvec_inputs_below_float16_precision_are_one_row(client, allow_authz):
    """
    On a halfvec collection, 1.0 and 1.0001 store identically and so are one embedding.

    Under the old md5-over-JSON-text hash these were different hashes, so both rows were
    written and the collection held two rows holding the exact same vector.
    """
    allow_authz("images")
    _create_collection(client, "images", vector_type="halfvec")

    first = client.post(
        "/vectorstore/collections/images/embeddings",
        json={"embeddings": [{"embedding": [1.0, 2.0, 3.0], "metadata": {}}]},
    )
    assert first.status_code == 200, first.text

    # float16 spacing near 1.0 is ~0.001, so this rounds onto the stored value above
    second = client.post(
        "/vectorstore/collections/images/embeddings",
        json={"embeddings": [{"embedding": [1.0001, 2.0, 3.0], "metadata": {}}]},
    )
    assert second.status_code == 409, second.text


def test_duplicates_within_one_request_collapse_to_one_row(client, allow_authz):
    """
    Repeats inside a single request write one row, and every input still gets a result.

    Callers index the response by position, so the deduplicated row is reported once per
    input that mapped onto it.
    """
    allow_authz("docs")
    _create_collection(client, "docs")

    response = client.post(
        "/vectorstore/collections/docs/embeddings",
        json={
            "embeddings": [
                {"embedding": [1.0, 2.0, 3.0], "metadata": {}},
                {"embedding": [4.0, 5.0, 6.0], "metadata": {}},
                {"embedding": [1.0, 2.0, 3.0], "metadata": {}},
            ]
        },
    )
    assert response.status_code == 200, response.text

    ids = [embedding["embedding_id"] for embedding in response.json()["embeddings"]]
    assert len(ids) == 3
    assert ids[0] == ids[2]
    assert ids[0] != ids[1]

    listed = client.get("/vectorstore/collections/docs/embeddings")
    assert len(listed.json()["embeddings"]) == 2


def test_results_map_to_the_input_that_produced_them(client, allow_authz):
    """
    Each returned embedding is the one its input asked for.

    Results are matched back by content hash rather than by the order Postgres returned rows
    in, which it does not promise.
    """
    allow_authz("docs")
    _create_collection(client, "docs")

    vectors = [[float(i), 0.0, 0.0] for i in range(1, 6)]
    response = client.post(
        "/vectorstore/collections/docs/embeddings",
        json={"embeddings": [{"embedding": vector, "metadata": {"i": index}} for index, vector in enumerate(vectors)]},
    )
    assert response.status_code == 200, response.text

    for index, embedding in enumerate(response.json()["embeddings"]):
        fetched = client.get(f"/vectorstore/collections/docs/embeddings/{embedding['embedding_id']}")
        assert fetched.status_code == 200, fetched.text
        assert fetched.json()["info"]["metadata"] == {"i": index}


def test_upsert_of_existing_content_updates_instead_of_inserting(client, allow_authz):
    """A PUT of content that already exists conflicts onto the existing row."""
    allow_authz("docs")
    _create_collection(client, "docs")

    body = {"embeddings": [{"embedding": [1.0, 2.0, 3.0], "metadata": {"a": 1}}]}
    created = client.post("/vectorstore/collections/docs/embeddings", json=body)
    assert created.status_code == 200, created.text
    created_id = created.json()["embeddings"][0]["embedding_id"]

    upserted = client.put("/vectorstore/collections/docs/embeddings", json=body)
    assert upserted.status_code == 200, upserted.text
    assert upserted.json()["embeddings"][0]["embedding_id"] == created_id

    listed = client.get("/vectorstore/collections/docs/embeddings")
    assert len(listed.json()["embeddings"]) == 1


def test_updating_a_row_onto_another_rows_content_conflicts(client, allow_authz):
    """
    Updating one row to duplicate another is a 409.

    The single-row update used to hash metadata as raw `json.dumps` text while bulk writes
    hashed Postgres's jsonb rendering, so the two paths could disagree about what a duplicate
    was. Both now go through the same canonical form.
    """
    allow_authz("docs")
    _create_collection(client, "docs")

    created = client.post(
        "/vectorstore/collections/docs/embeddings",
        json={
            "embeddings": [
                {"embedding": [1.0, 2.0, 3.0], "metadata": {"a": 1, "b": 2}},
                {"embedding": [4.0, 5.0, 6.0], "metadata": {"a": 1, "b": 2}},
            ]
        },
    )
    assert created.status_code == 200, created.text
    second_id = created.json()["embeddings"][1]["embedding_id"]

    collide = client.put(
        "/vectorstore/collections/docs/embeddings",
        json={
            "embeddings": [
                {"embedding_id": second_id, "embedding": [1.0, 2.0, 3.0], "metadata": {"b": 2, "a": 1}},
            ]
        },
    )
    assert collide.status_code == 409, collide.text


def test_dimension_mismatch_is_rejected_before_the_write(client, allow_authz):
    """
    A wrong-length vector is a 400.

    The route checks this too, but the check is repeated in the data access layer because the
    bulk INSERT binds the batch as one flat array sliced by the collection's dimensionality:
    a wrong length there would silently shift every following row rather than fail.
    """
    allow_authz("docs")
    _create_collection(client, "docs")

    response = client.post(
        "/vectorstore/collections/docs/embeddings",
        json={"embeddings": [{"embedding": [1.0, 2.0], "metadata": {}}]},
    )
    assert response.status_code == 400, response.text
