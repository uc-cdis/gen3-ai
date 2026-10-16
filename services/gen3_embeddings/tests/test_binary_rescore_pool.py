"""
Binary-index search at the largest `top_k`, against a real pgvector.

At `MAX_TOP_K` the rescore pool is wider than the largest `hnsw.ef_search` pgvector accepts, so
these check the two things only Postgres can report: that setting up the search does not fail,
and how many candidates the index actually hands to the rescore.
"""

import json

import asyncpg
import numpy as np
import pytest
from pgvector.asyncpg import register_vector

from gen3_embeddings import config
from gen3_embeddings.database.db import DataAccessLayer
from gen3_embeddings.database.helpers import build_search_sql
from gen3_embeddings.database.index_discovery import discover_vector_indexes
from gen3_embeddings.models.schemas import DistanceMetric, VectorType

COLLECTION = "binary_top_k"
AUTHZ = f"/vectorstore/collections/{COLLECTION}"
DIMENSIONS = 64
# More rows than the widest pool, so a pool that comes back short was cut by the index scan
ROWS = 3_000
METRIC = DistanceMetric.cosine_similarity


async def _seed_binary_indexed_collection(test_database) -> tuple[int, str]:
    """
    Write ROWS embeddings through the DAL, then build the README's binary index over them.

    Returns:
        tuple[int, str]: The collection id and the index name.
    """
    pool = await asyncpg.create_pool(test_database["app_dsn"], min_size=1, max_size=1, init=register_vector)
    try:
        dal = DataAccessLayer(pool, allowed_authz=[AUTHZ], allowed_collection_names={COLLECTION})
        collection = await dal.create_collection(COLLECTION, None, DIMENSIONS)
        vectors = np.random.default_rng(0).uniform(-1, 1, (ROWS, DIMENSIONS)).tolist()
        for start in range(0, ROWS, config.MAX_EMBEDDINGS_PER_REQUEST):
            batch = vectors[start : start + config.MAX_EMBEDDINGS_PER_REQUEST]
            await dal.create_embeddings_bulk(collection, batch, AUTHZ, None)
    finally:
        await pool.close()

    index_name = f"idx_binary_top_k_c{collection.id}"
    admin = await asyncpg.connect(test_database["admin_dsn"])
    try:
        await admin.execute(
            f"CREATE INDEX {index_name} ON embeddings_vector "
            f"USING hnsw ((binary_quantize(embedding)::bit({DIMENSIONS})) bit_hamming_ops) "
            f"WHERE collection_id = {collection.id}"
        )
        await admin.execute("ANALYZE embeddings_vector")
    finally:
        await admin.close()
    return collection.id, index_name


def _find_node(node: dict, key: str, value: str) -> dict | None:
    """The first plan node, depth first, whose `key` is `value`."""
    if node.get(key) == value:
        return node
    for child in node.get("Plans", []):
        found = _find_node(child, key, value)
        if found is not None:
            return found
    return None


def _plan_index_names(node: dict) -> set[str]:
    """Every index a plan node or any node beneath it reads."""
    names = {node["Index Name"]} if "Index Name" in node else set()
    for child in node.get("Plans", []):
        names |= _plan_index_names(child)
    return names


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("iterative_scan", "expected_candidates"),
    [
        # the scan keeps going past ef_search until the pool's LIMIT is met
        ("relaxed_order", config.BINARY_RESCORE_MAX),
        # the index stops at ef_search, which is capped
        ("off", config.HNSW_MAX_EF_SEARCH),
    ],
)
async def test_binary_search_at_max_top_k(test_database, reset_db, monkeypatch, iterative_scan, expected_candidates):
    """
    A binary-indexed search at MAX_TOP_K returns top_k rows, from as many candidates as the scan mode allows.

    The pool at MAX_TOP_K is BINARY_RESCORE_MAX, which is wider than HNSW_MAX_EF_SEARCH. Raising
    `hnsw.ef_search` to the pool itself is an error, so without the cap this search fails
    before it runs.
    """
    assert config.BINARY_RESCORE_MAX > config.HNSW_MAX_EF_SEARCH, "the pool must exceed the cap to test it"
    assert expected_candidates >= config.MAX_TOP_K
    monkeypatch.setattr(config, "HNSW_ITERATIVE_SCAN", iterative_scan)

    collection_id, index_name = await _seed_binary_indexed_collection(test_database)

    # On a table this small the planner reaches the index only with its alternatives disabled;
    # see TestSearchSqlUsesTheIndex in test_index_discovery.py. Set at connection start, so the
    # pool's RESET ALL keeps them.
    pool = await asyncpg.create_pool(
        test_database["app_dsn"],
        min_size=1,
        max_size=1,
        init=register_vector,
        server_settings={"enable_seqscan": "off", "enable_sort": "off"},
    )
    try:
        async with pool.acquire() as conn:
            vector_indexes = await discover_vector_indexes(conn)
        dal = DataAccessLayer(
            pool, allowed_authz=[AUTHZ], allowed_collection_names={COLLECTION}, vector_indexes=vector_indexes
        )
        collection = await dal.get_collection_by_name(COLLECTION)
        assert collection is not None and collection.id == collection_id

        query = np.random.default_rng(1).uniform(-1, 1, DIMENSIONS).tolist()
        top_k = config.MAX_TOP_K
        rescore_limit = dal._binary_rescore_limit(collection_id, "embeddings_vector", METRIC, DIMENSIONS, top_k)
        assert rescore_limit == config.BINARY_RESCORE_MAX

        rows = await dal.search_embeddings_in_collection(
            collection, query, top_k=top_k, min_value=None, max_value=None, distance_metric=METRIC, filters=None
        )
        assert len(rows) == top_k

        # The same statement the search ran, with its real parameters, to see what the index did
        sql, extra = build_search_sql(
            table="embeddings_vector",
            distance_metric=METRIC,
            collection_id_param="$1::bigint",
            vector_placeholder="$2",
            top_k_param="$3::int",
            filters=None,
            min_value=None,
            max_value=None,
            vector_type=VectorType.vector,
            dimensions=DIMENSIONS,
            binary_rescore_limit=rescore_limit,
        )

        async def _explain(conn):
            return await conn.fetchval(f"EXPLAIN (ANALYZE, FORMAT JSON) {sql}", collection_id, query, top_k, *extra)

        plan = json.loads(await dal._with_rls(_explain, ef_search=rescore_limit))[0]["Plan"]
    finally:
        await pool.close()
        admin = await asyncpg.connect(test_database["admin_dsn"])
        try:
            await admin.execute(f"DROP INDEX IF EXISTS {index_name}")
        finally:
            await admin.close()

    assert index_name in _plan_index_names(plan), "the candidate stage did not read the binary index"
    candidates = _find_node(plan, "Subplan Name", "CTE candidates")
    assert candidates is not None, "no candidates CTE in the plan"
    assert round(candidates["Actual Rows"]) == expected_candidates
