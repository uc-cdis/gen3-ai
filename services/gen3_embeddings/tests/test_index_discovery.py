"""
Tests for reading the vector indexes that actually exist (database/index_discovery.py).

Search has to emit the exact expression an index was built on, so choosing the wrong shape is
not a small error: it is a silent fall back to a sequential scan, which at 15M rows a
collection is the difference between milliseconds and minutes. Everything here is about
getting that choice right, and about failing towards the slow-but-correct answer when the
catalog holds something this module was not written to understand.
"""

import asyncio

import asyncpg
import pytest

from gen3_embeddings.database.index_discovery import (
    IndexStrategy,
    VectorIndex,
    _parse,
    choose_index,
    discover_vector_indexes,
)
from gen3_embeddings.models.schemas import DistanceMetric, VectorType


def row(**overrides):
    """A row of DISCOVERY_SQL, defaulting to the shape the index script produces."""
    base = {
        "index_name": "idx_ev_hnsw_c1_cos",
        "table_name": "embeddings_vector",
        "opclass": "vector_cosine_ops",
        "expression": "(embedding)::vector(256)",
        "predicate": "(collection_id = 1)",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# _parse
# ---------------------------------------------------------------------------


def test_a_direct_index_is_parsed_with_its_metric_and_type():
    """A cosine index over `vector` can answer cosine orderings for that collection."""
    found = _parse("embeddings_vector", row())

    assert found == VectorIndex(
        index_name="idx_ev_hnsw_c1_cos",
        table="embeddings_vector",
        collection_id=1,
        strategy=IndexStrategy.direct,
        dimensions=256,
        metric=DistanceMetric.cosine_distance,
        vector_type=VectorType.vector,
    )


def test_a_binary_index_is_not_tied_to_a_metric():
    """Hamming only selects candidates; the rescore ranks them by whatever was requested."""
    found = _parse(
        "embeddings_vector",
        row(opclass="bit_hamming_ops", expression="(binary_quantize(embedding))::bit(1536)"),
    )

    assert found.strategy is IndexStrategy.binary
    assert found.dimensions == 1536
    assert found.metric is None
    assert found.vector_type is None


def test_a_halfvec_index_reports_its_storage_type():
    """The cast search emits has to match, and halfvec and vector are different casts."""
    found = _parse("embeddings_halfvec", row(opclass="halfvec_l2_ops", expression="(embedding)::halfvec(768)"))

    assert (found.vector_type, found.metric, found.dimensions) == (
        VectorType.halfvec,
        DistanceMetric.l2_distance,
        768,
    )


@pytest.mark.parametrize(
    "predicate",
    [
        "(collection_id = ANY (ARRAY[1, 2]))",
        "((collection_id = 1) AND (authz = 'x'::text))",
        "(dimensions = 1)",
        None,
    ],
)
def test_a_predicate_this_module_does_not_understand_is_skipped(predicate):
    """
    Anything but the simple per-collection predicate is left undiscovered, on purpose.

    Search then emits its unindexed query, which is slow but returns the right rows. Guessing
    that a more complex predicate covers the whole collection would emit SQL the index cannot
    serve, and Postgres reports nothing when an expression fails to match.
    """
    assert _parse("embeddings_vector", row(predicate=predicate)) is None


def test_an_unsupported_opclass_is_skipped():
    """A jaccard index exists in pgvector but search has no shape that targets it."""
    assert _parse("embeddings_vector", row(opclass="bit_jaccard_ops")) is None


# ---------------------------------------------------------------------------
# choose_index
# ---------------------------------------------------------------------------

DIRECT_COSINE = VectorIndex(
    "i_direct", "embeddings_vector", 1, IndexStrategy.direct, 256, DistanceMetric.cosine_distance, VectorType.vector
)
BINARY = VectorIndex("i_binary", "embeddings_vector", 1, IndexStrategy.binary, 256, None, None)


def test_no_index_means_no_index():
    """An unindexed collection must produce the plain query rather than a guess."""
    assert choose_index(None, DistanceMetric.cosine_similarity, 256) is None
    assert choose_index([], DistanceMetric.cosine_similarity, 256) is None


def test_cosine_similarity_is_served_by_a_cosine_distance_index():
    """The two share an ordering: search orders by the distance and reports `1 - distance`."""
    assert choose_index([DIRECT_COSINE], DistanceMetric.cosine_similarity, 256) is DIRECT_COSINE


def test_a_direct_index_does_not_serve_a_different_metric():
    """pgvector will not answer an l2 ordering from a cosine index, so this must fall back."""
    assert choose_index([DIRECT_COSINE], DistanceMetric.l2_distance, 256) is None


def test_a_binary_index_serves_any_metric():
    """It only picks candidates, so the metric is decided entirely by the rescore."""
    for metric in DistanceMetric:
        assert choose_index([BINARY], metric, 256) is BINARY


def test_a_direct_index_wins_over_a_binary_one():
    """No rescore step and no precision lost, so prefer it whenever it can serve the metric."""
    assert choose_index([BINARY, DIRECT_COSINE], DistanceMetric.cosine_distance, 256) is DIRECT_COSINE


def test_binary_is_used_when_direct_cannot_serve_the_metric():
    """Having both, an l2 query still has somewhere to go."""
    assert choose_index([BINARY, DIRECT_COSINE], DistanceMetric.l2_distance, 256) is BINARY


def test_an_index_of_the_wrong_dimension_is_never_chosen():
    """
    A mismatched cast cannot use the index, and would also be wrong.

    This is the case where a collection was recreated at a different dimensionality and the
    old index is still sitting there.
    """
    assert choose_index([DIRECT_COSINE, BINARY], DistanceMetric.cosine_distance, 1536) is None


# ---------------------------------------------------------------------------
# discover_vector_indexes, against a real catalog
# ---------------------------------------------------------------------------


def test_discovery_finds_a_real_index_and_ignores_an_invalid_one(test_database):
    """
    End to end against Postgres, including the invalid-index case seen in practice.

    A failed `CREATE INDEX CONCURRENTLY` leaves an INVALID index behind that Postgres will
    never use. Reporting one as usable would have search emit an expression nothing can serve.
    """

    async def _run():
        conn = await asyncpg.connect(test_database["admin_dsn"])
        try:
            collection_id = await conn.fetchval(
                "INSERT INTO collections (collection_name, dimensions, vector_type) "
                "VALUES ('idx_disc', 3, 'vector') RETURNING id"
            )
            await conn.execute(
                f"CREATE INDEX idx_disc_cos ON embeddings_vector "
                f"USING hnsw ((embedding::vector(3)) vector_cosine_ops) WHERE collection_id = {collection_id}"
            )
            # mark it invalid the way a cancelled CONCURRENTLY build leaves it
            await conn.execute(
                f"CREATE INDEX idx_disc_dead ON embeddings_vector "
                f"USING hnsw ((embedding::vector(3)) vector_l2_ops) WHERE collection_id = {collection_id}"
            )
            await conn.execute("UPDATE pg_index SET indisvalid = false WHERE indexrelid = 'idx_disc_dead'::regclass")

            return collection_id, await discover_vector_indexes(conn)
        finally:
            await conn.close()

    collection_id, found = asyncio.run(_run())

    names = [index.index_name for index in found[collection_id]]
    assert names == ["idx_disc_cos"]
    assert found[collection_id][0].table == "embeddings_vector"
    assert found[collection_id][0].strategy is IndexStrategy.direct
    # the l2 index is invalid, so an l2 search must not be pointed at it
    assert choose_index(found[collection_id], DistanceMetric.l2_distance, 3) is None
