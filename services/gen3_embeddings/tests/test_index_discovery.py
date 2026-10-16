"""
Tests for reading the vector indexes that actually exist (database/index_discovery.py).

Search has to emit the exact expression an index was built on, so choosing the wrong shape is
not a small error: it is a silent fall back to a sequential scan, which at 15M rows a
collection is the difference between milliseconds and minutes. Everything here is about
getting that choice right, and about failing towards the slow-but-correct answer when the
catalog holds something this module was not written to understand.
"""

import asyncio
import json

import asyncpg
import pytest
from pgvector.asyncpg import register_vector

from gen3_embeddings.database.db import DataAccessLayer
from gen3_embeddings.database.helpers import (
    CollectionSearchTarget,
    build_multi_collection_search_sql,
    build_search_sql,
)
from gen3_embeddings.database.index_discovery import (
    IndexStrategy,
    VectorIndex,
    _parse,
    choose_index,
    discover_vector_indexes,
    find_collection_index_names,
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


class TestParse:
    """Turning a catalog row into a VectorIndex, or skipping one this module cannot use."""

    def test_a_direct_index_is_parsed_with_its_metric_and_type(self):
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

    def test_a_binary_index_is_not_tied_to_a_metric(self):
        """Hamming only selects candidates; the rescore ranks them by whatever was requested."""
        found = _parse(
            "embeddings_vector",
            row(opclass="bit_hamming_ops", expression="(binary_quantize(embedding))::bit(1536)"),
        )

        assert found is not None
        assert found.strategy is IndexStrategy.binary
        assert found.dimensions == 1536
        assert found.metric is None
        assert found.vector_type is None

    def test_a_halfvec_index_reports_its_storage_type(self):
        """The cast search emits has to match, and halfvec and vector are different casts."""
        found = _parse("embeddings_halfvec", row(opclass="halfvec_l2_ops", expression="(embedding)::halfvec(768)"))

        assert found is not None
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
    def test_a_predicate_this_module_does_not_understand_is_skipped(self, predicate):
        """
        Anything but the simple per-collection predicate is left undiscovered, on purpose.

        Search then emits its unindexed query, which is slow but returns the right rows. Guessing
        that a more complex predicate covers the whole collection would emit SQL the index cannot
        serve, and Postgres reports nothing when an expression fails to match.
        """
        assert _parse("embeddings_vector", row(predicate=predicate)) is None

    def test_an_unsupported_opclass_is_skipped(self):
        """A jaccard index exists in pgvector but search has no shape that targets it."""
        assert _parse("embeddings_vector", row(opclass="bit_jaccard_ops")) is None

    @pytest.mark.parametrize(
        ("table", "opclass", "expression"),
        [
            # cast before quantizing: reads naturally, but search quantizes the bare column
            ("embeddings_vector", "bit_hamming_ops", "(binary_quantize((embedding)::vector(3)))::bit(3)"),
            # search casts a `vector` table's column to `vector(n)`, never `halfvec(n)`
            ("embeddings_vector", "halfvec_cosine_ops", "(embedding)::halfvec(3)"),
            ("embeddings_halfvec", "vector_cosine_ops", "(embedding)::vector(3)"),
            ("embeddings_vector", "vector_cosine_ops", "(l2_normalize(embedding))::vector(3)"),
        ],
    )
    def test_an_expression_search_does_not_emit_is_skipped(self, table, opclass, expression):
        """
        Only the exact expression search writes counts, not anything ending in a width.

        Each of these is valid DDL, and reading only its trailing `(n)` would have search pick
        the index and then emit an expression it does not match: a sequential scan.
        """
        assert _parse(table, row(table_name=table, opclass=opclass, expression=expression)) is None


TABLE = "embeddings_vector"
DIRECT_COSINE = VectorIndex(
    "i_direct", TABLE, 1, IndexStrategy.direct, 256, DistanceMetric.cosine_distance, VectorType.vector
)
BINARY = VectorIndex("i_binary", TABLE, 1, IndexStrategy.binary, 256, None, None)


class TestChooseIndex:
    """Picking which known index, if any, can serve a search."""

    def test_no_index_means_no_index(self):
        """An unindexed collection must produce the plain query rather than a guess."""
        assert choose_index(None, TABLE, DistanceMetric.cosine_similarity, 256) is None
        assert choose_index([], TABLE, DistanceMetric.cosine_similarity, 256) is None

    def test_cosine_similarity_is_served_by_a_cosine_distance_index(self):
        """The two share an ordering: search orders by the distance and reports `1 - distance`."""
        assert choose_index([DIRECT_COSINE], TABLE, DistanceMetric.cosine_similarity, 256) is DIRECT_COSINE

    def test_a_direct_index_does_not_serve_a_different_metric(self):
        """pgvector will not answer an l2 ordering from a cosine index, so this must fall back."""
        assert choose_index([DIRECT_COSINE], TABLE, DistanceMetric.l2_distance, 256) is None

    def test_a_binary_index_serves_any_metric(self):
        """It only picks candidates, so the metric is decided entirely by the rescore."""
        for metric in DistanceMetric:
            assert choose_index([BINARY], TABLE, metric, 256) is BINARY

    def test_a_direct_index_wins_over_a_binary_one(self):
        """No rescore step and no precision lost, so prefer it whenever it can serve the metric."""
        assert choose_index([BINARY, DIRECT_COSINE], TABLE, DistanceMetric.cosine_distance, 256) is DIRECT_COSINE

    def test_binary_is_used_when_direct_cannot_serve_the_metric(self):
        """Having both, an l2 query still has somewhere to go."""
        assert choose_index([BINARY, DIRECT_COSINE], TABLE, DistanceMetric.l2_distance, 256) is BINARY

    def test_an_index_of_the_wrong_dimension_is_never_chosen(self):
        """
        A mismatched cast cannot use the index, and would also be wrong.

        This is the case where a collection was recreated at a different dimensionality and the
        old index is still sitting there.
        """
        assert choose_index([DIRECT_COSINE, BINARY], TABLE, DistanceMetric.cosine_distance, 1536) is None

    def test_an_index_on_the_other_table_is_never_chosen(self):
        """
        An index is keyed by the collection id in its predicate, which says nothing of its table.

        One built on `embeddings_halfvec` for a `vector` collection indexes rows search never reads.
        """
        assert choose_index([DIRECT_COSINE, BINARY], "embeddings_halfvec", DistanceMetric.cosine_distance, 256) is None


class TestDiscoverVectorIndexes:
    """Discovery against a real Postgres catalog."""

    def test_discovery_finds_a_real_index_and_ignores_an_invalid_one(self, test_database):
        """
        End to end against Postgres, including an INVALID index left by a failed concurrent build.

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
                await conn.execute(
                    "UPDATE pg_index SET indisvalid = false WHERE indexrelid = 'idx_disc_dead'::regclass"
                )

                return collection_id, await discover_vector_indexes(conn)
            finally:
                await conn.close()

        collection_id, found = asyncio.run(_run())

        names = [index.index_name for index in found[collection_id]]
        assert names == ["idx_disc_cos"]
        assert found[collection_id][0].table == "embeddings_vector"
        assert found[collection_id][0].strategy is IndexStrategy.direct
        # the l2 index is invalid, so an l2 search must not be pointed at it
        assert choose_index(found[collection_id], TABLE, DistanceMetric.l2_distance, 3) is None

    def test_indexes_search_cannot_use_are_never_chosen(self, test_database):
        """
        Near misses of the README shapes, built for real, leave search on the unindexed query.

        Postgres accepts every one of these, and none matches the SQL search emits for a
        `vector` collection, so choosing any of them would plan a sequential scan.
        """

        async def _run():
            conn = await asyncpg.connect(test_database["admin_dsn"])
            try:
                collection_id = await conn.fetchval(
                    "INSERT INTO collections (collection_name, dimensions, vector_type) "
                    "VALUES ('idx_near_miss', 3, 'vector') RETURNING id"
                )
                await conn.execute(
                    f"CREATE INDEX idx_miss_cast_first ON embeddings_vector "
                    f"USING hnsw ((binary_quantize(embedding::vector(3))::bit(3)) bit_hamming_ops) "
                    f"WHERE collection_id = {collection_id}"
                )
                await conn.execute(
                    f"CREATE INDEX idx_miss_halfvec_cast ON embeddings_vector "
                    f"USING hnsw ((embedding::halfvec(3)) halfvec_cosine_ops) WHERE collection_id = {collection_id}"
                )
                await conn.execute(
                    f"CREATE INDEX idx_miss_other_table ON embeddings_halfvec "
                    f"USING hnsw ((embedding::halfvec(3)) halfvec_cosine_ops) WHERE collection_id = {collection_id}"
                )

                return collection_id, await discover_vector_indexes(conn)
            finally:
                await conn.close()

        collection_id, found = asyncio.run(_run())

        # the other-table index is a shape search does emit, just not for this collection
        assert [index.index_name for index in found.get(collection_id, [])] == ["idx_miss_other_table"]
        for metric in DistanceMetric:
            assert choose_index(found.get(collection_id), TABLE, metric, 3) is None

    def test_a_collections_index_names_include_unusable_ones_and_no_other_collections(self, test_database):
        """
        Listing what is left to drop covers both tables and keeps the INVALID index discovery skips.

        A build that fails late leaves an INVALID index Postgres still maintains, so hiding it
        would hide exactly the index most likely to be forgotten.
        """

        async def _run():
            conn = await asyncpg.connect(test_database["admin_dsn"])
            try:
                ids = []
                for name in ("idx_find_a", "idx_find_b"):
                    ids.append(
                        await conn.fetchval(
                            "INSERT INTO collections (collection_name, dimensions, vector_type) "
                            "VALUES ($1, 3, 'vector') RETURNING id",
                            name,
                        )
                    )
                a, b = ids
                await conn.execute(
                    f"CREATE INDEX idx_find_a_cos ON embeddings_vector "
                    f"USING hnsw ((embedding::vector(3)) vector_cosine_ops) WHERE collection_id = {a}"
                )
                await conn.execute(
                    f"CREATE INDEX idx_find_a_dead ON embeddings_vector "
                    f"USING hnsw ((embedding::vector(3)) vector_l2_ops) WHERE collection_id = {a}"
                )
                await conn.execute(
                    "UPDATE pg_index SET indisvalid = false WHERE indexrelid = 'idx_find_a_dead'::regclass"
                )
                await conn.execute(
                    f"CREATE INDEX idx_find_a_half ON embeddings_halfvec "
                    f"USING hnsw ((embedding::halfvec(3)) halfvec_l2_ops) WHERE collection_id = {a}"
                )
                await conn.execute(
                    f"CREATE INDEX idx_find_b_cos ON embeddings_vector "
                    f"USING hnsw ((embedding::vector(3)) vector_cosine_ops) WHERE collection_id = {b}"
                )
            finally:
                await conn.close()

            # delete_collection runs this as the app role, which owns nothing and bypasses no RLS
            app_conn = await asyncpg.connect(test_database["app_dsn"])
            try:
                return await find_collection_index_names(app_conn, a)
            finally:
                await app_conn.close()

        assert asyncio.run(_run()) == ["idx_find_a_cos", "idx_find_a_dead", "idx_find_a_half"]


# The two index shapes the index migration README documents, with the width and collection
# left to fill in. Search has to match these as an operator would write them, not as our own
# SQL builder would.
README_INDEX_DDL = {
    IndexStrategy.direct: (
        "CREATE INDEX {name} ON embeddings_vector "
        "USING hnsw ((embedding::vector({dimensions})) vector_cosine_ops) "
        "WHERE collection_id = {collection_id}"
    ),
    IndexStrategy.binary: (
        "CREATE INDEX {name} ON embeddings_vector "
        "USING hnsw ((binary_quantize(embedding)::bit({dimensions})) bit_hamming_ops) "
        "WHERE collection_id = {collection_id}"
    ),
}

PLAN_DIMENSIONS = 3
PLAN_QUERY = [1.0, 0.0, 0.0]
PLAN_TOP_K = 5
# The API's default metric, and the one whose reported value is not what the index orders by.
PLAN_METRIC = DistanceMetric.cosine_similarity


def plan_index_names(node: dict) -> set[str]:
    """Every index a plan node or any node beneath it reads, CTEs and subqueries included."""
    names = {node["Index Name"]} if "Index Name" in node else set()
    for child in node.get("Plans", []):
        names |= plan_index_names(child)
    return names


async def explain_search(test_database, strategy: IndexStrategy, shape: str) -> tuple[set[str], set[str]]:
    """
    Index one collection per arm the README's way, then EXPLAIN the search the service would run.

    Discovery, `choose_index` and the rescore limit all come from the service's own code, so a
    README index that discovery misreads fails here the same way it would in production.

    Args:
        test_database: The session database fixture.
        strategy (IndexStrategy): Which README index to build.
        shape (str): "single" for `build_search_sql`, "multi" for a two-arm
            `build_multi_collection_search_sql`.

    Returns:
        tuple[set[str], set[str]]: The indexes built, and the indexes the plan reads.
    """
    admin = await asyncpg.connect(test_database["admin_dsn"])
    try:
        built: dict[int, str] = {}
        for arm in range(1 if shape == "single" else 2):
            collection_id = await admin.fetchval(
                "INSERT INTO collections (collection_name, dimensions, vector_type) "
                "VALUES ($1, $2, 'vector') RETURNING id",
                f"plan_{strategy.value}_{shape}_{arm}",
                PLAN_DIMENSIONS,
            )
            name = f"idx_plan_c{collection_id}_{strategy.value}"
            await admin.execute(
                README_INDEX_DDL[strategy].format(name=name, dimensions=PLAN_DIMENSIONS, collection_id=collection_id)
            )
            built[collection_id] = name
    finally:
        await admin.close()

    pool = await asyncpg.create_pool(test_database["app_dsn"], min_size=1, max_size=1, init=register_vector)
    try:
        async with pool.acquire() as conn:
            found = await discover_vector_indexes(conn)
        dal = DataAccessLayer(pool, allowed_authz=["/plan"], vector_indexes=found)

        # the same arguments `search_embeddings_in_collection` and
        # `search_embeddings_across_collections` pass
        targets = [
            CollectionSearchTarget(
                collection_id, dal._binary_rescore_limit(collection_id, TABLE, PLAN_METRIC, PLAN_DIMENSIONS, PLAN_TOP_K)
            )
            for collection_id in built
        ]
        if shape == "single":
            sql, extra = build_search_sql(
                table="embeddings_vector",
                distance_metric=PLAN_METRIC,
                collection_id_param="$1::bigint",
                vector_placeholder="$2",
                top_k_param="$3::int",
                filters=None,
                min_value=None,
                max_value=None,
                vector_type=VectorType.vector,
                dimensions=PLAN_DIMENSIONS,
                binary_rescore_limit=targets[0].binary_rescore_limit,
            )
            params = [targets[0].collection_id, PLAN_QUERY, PLAN_TOP_K, *extra]
        else:
            sql, extra = build_multi_collection_search_sql(
                table="embeddings_vector",
                distance_metric=PLAN_METRIC,
                targets=targets,
                vector_placeholder="$1",
                top_k_param="$2::int",
                filters=None,
                min_value=None,
                max_value=None,
                vector_type=VectorType.vector,
                dimensions=PLAN_DIMENSIONS,
            )
            params = [PLAN_QUERY, PLAN_TOP_K, *extra]

        # a binary target is the only kind with a rescore limit, so this is the strategy chosen
        chosen = {IndexStrategy.binary if t.binary_rescore_limit else IndexStrategy.direct for t in targets}
        assert chosen == {strategy}, f"discovery chose {chosen} for a README {strategy.value} index"

        async def _explain(conn):
            await conn.execute("SET LOCAL enable_seqscan = off")
            await conn.execute("SET LOCAL enable_sort = off")
            return await conn.fetchval(f"EXPLAIN (FORMAT JSON) {sql}", *params)

        rescore_limits = [t.binary_rescore_limit for t in targets if t.binary_rescore_limit is not None]
        plan = await dal._with_rls(_explain, ef_search=max(rescore_limits, default=None))
    finally:
        await pool.close()

    return set(built.values()), plan_index_names(json.loads(plan)[0]["Plan"])


class TestSearchSqlUsesTheIndex:
    """
    The emitted search SQL, judged by whether Postgres actually reads the index for it.

    A query that stops matching its index is still valid SQL, so nothing errors: the plan just
    becomes a scan. Only the planner can report that, which is why these EXPLAIN the statement
    with its real parameters bound, under the same transaction settings `_with_rls` applies,
    against an index built the way the README tells operators to build one.

    On a table this small the planner prefers not to touch the vector index whether or not
    the query matches it, so two settings take its alternatives away. `enable_seqscan` rules
    out the scan; `enable_sort` rules out the next cheapest plan, which narrows to one
    collection on the `(collection_id, content hash)` btree and sorts what it finds. That
    leaves an ordered vector index scan as the only plan left, so the index shows up in the
    plan exactly when the query can use it: drop the cast or change its width and the btree
    plan comes back.
    """

    @pytest.mark.parametrize("shape", ["single", "multi"])
    def test_a_direct_cosine_index_serves_the_search(self, test_database, shape):
        """The ordering expression has to be the index's own cast and operator, with no wrapper."""
        built, read = asyncio.run(explain_search(test_database, IndexStrategy.direct, shape))

        assert built <= read, f"plan reads {read or 'no index'}, expected {built}"

    @pytest.mark.parametrize("shape", ["single", "multi"])
    def test_a_binary_index_serves_the_candidate_stage(self, test_database, shape):
        """The Hamming ordering has to match `binary_quantize(embedding)::bit(n)` as the README builds it."""
        built, read = asyncio.run(explain_search(test_database, IndexStrategy.binary, shape))

        assert built <= read, f"plan reads {read or 'no index'}, expected {built}"
