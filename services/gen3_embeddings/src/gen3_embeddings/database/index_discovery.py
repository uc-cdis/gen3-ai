"""
Ask Postgres which vector indexes exist, so search can emit SQL that one of them can serve.

The `embedding` columns carry no dimension (`VECTOR`, not `VECTOR(n)`), because one table
holds every collection and collections differ in dimensionality. pgvector cannot index a
dimensionless column, so the only indexable form is a partial expression index per
collection, and the query has to reproduce that expression *exactly* or the plan silently
falls back to a sequential scan. Which expression to emit therefore depends on which index
is actually there.

Nothing else in the service knows that. Index creation is deliberately an operator action --
a script or a migration, never defined from a request -- so the app cannot learn it as a side
effect of having built one. Reading `pg_index` is the one source that cannot drift: a column
on `collections` recording "this one is binary quantized" stays behind when someone drops the
index, and the app then emits binary-quantized SQL against no such index, which is a full
scan of the whole collection rather than a millisecond lookup.

Requires no privileges beyond connecting: Postgres grants SELECT on these catalogs to PUBLIC,
and `pg_get_expr` performs no table-level check. Verified against a role holding nothing but
LOGIN and USAGE on the schema.
"""

import re
from dataclasses import dataclass
from enum import StrEnum

import asyncpg

from gen3_embeddings.config import logging
from gen3_embeddings.models.schemas import DistanceMetric, VectorType

# Only the simple predicate the index scripts and documented migrations produce. A predicate
# this does not match (`collection_id IN (...)`, an extra AND term) is left undiscovered, so
# search falls back to the unindexed query -- slow but correct, which is the right way to fail.
_PREDICATE = re.compile(r"^\(collection_id = (\d+)\)$")

# `(embedding)::vector(256)` / `(binary_quantize(embedding))::bit(1536)`
_DIMENSIONS = re.compile(r"\((\d+)\)$")

# opclass -> the metric it can order by. A direct index serves ONLY its own metric; pgvector
# will not answer an l2 ordering from a cosine index.
_OPCLASS_METRIC: dict[str, DistanceMetric] = {
    "vector_cosine_ops": DistanceMetric.cosine_distance,
    "vector_l2_ops": DistanceMetric.l2_distance,
    "vector_ip_ops": DistanceMetric.inner_product,
    "vector_l1_ops": DistanceMetric.l1_distance,
    "halfvec_cosine_ops": DistanceMetric.cosine_distance,
    "halfvec_l2_ops": DistanceMetric.l2_distance,
    "halfvec_ip_ops": DistanceMetric.inner_product,
    "halfvec_l1_ops": DistanceMetric.l1_distance,
}

_OPCLASS_VECTOR_TYPE: dict[str, VectorType] = {
    "vector": VectorType.vector,
    "halfvec": VectorType.halfvec,
}

# Hamming distance over binary_quantize(). Unlike the direct opclasses this one is not tied to
# a metric: it only selects candidates, and the rescore step then ranks them by whatever
# metric the caller asked for. So one of these serves every metric.
_BINARY_OPCLASS = "bit_hamming_ops"

DISCOVERY_SQL = """
    SELECT i.relname                           AS index_name,
           t.relname                           AS table_name,
           opc.opcname                         AS opclass,
           pg_get_expr(x.indexprs, x.indrelid) AS expression,
           pg_get_expr(x.indpred, x.indrelid)  AS predicate
    FROM pg_index x
    JOIN pg_class i ON i.oid = x.indexrelid
    JOIN pg_class t ON t.oid = x.indrelid
    JOIN pg_am am ON am.oid = i.relam
    JOIN pg_opclass opc ON opc.oid = x.indclass[0]
    JOIN pg_namespace n ON n.oid = t.relnamespace
    WHERE n.nspname = 'public'
      AND t.relname = ANY($1::text[])
      AND am.amname IN ('hnsw', 'ivfflat')
      -- a failed CREATE INDEX CONCURRENTLY leaves an INVALID index behind, which Postgres
      -- will not use; treating one as usable would mean emitting SQL nothing can serve
      AND x.indisvalid
      AND x.indpred IS NOT NULL
      AND x.indexprs IS NOT NULL
"""


class IndexStrategy(StrEnum):
    """Which SQL shape a collection's index can answer."""

    # ORDER BY embedding::TYPE(n) <op> $q
    direct = "direct"
    # ORDER BY binary_quantize(embedding)::bit(n) <~> binary_quantize($q), then rescore exactly
    binary = "binary"


@dataclass(frozen=True)
class VectorIndex:
    """One usable per-collection vector index, as found in the catalog."""

    index_name: str
    table: str
    collection_id: int
    strategy: IndexStrategy
    dimensions: int
    # None for a binary index, which is not tied to a metric
    metric: DistanceMetric | None
    # None for a binary index, whose indexed expression is a bit string
    vector_type: VectorType | None


def _parse(table: str, row: asyncpg.Record) -> VectorIndex | None:
    """
    Turn one catalog row into a `VectorIndex`, or None if it is not a shape search can use.

    Args:
        table (str): Table the index belongs to.
        row (asyncpg.Record): A row of `DISCOVERY_SQL`.

    Returns:
        VectorIndex | None: The parsed index, or None to leave it undiscovered.
    """
    predicate = _PREDICATE.match((row["predicate"] or "").strip())
    dimensions = _DIMENSIONS.search((row["expression"] or "").strip())
    if not predicate or not dimensions:
        logging.debug(
            "Ignoring vector index %s: predicate %r / expression %r is not a shape search emits",
            row["index_name"],
            row["predicate"],
            row["expression"],
        )
        return None

    opclass = row["opclass"]
    if opclass == _BINARY_OPCLASS:
        strategy, metric, vector_type = IndexStrategy.binary, None, None
    elif opclass in _OPCLASS_METRIC:
        strategy = IndexStrategy.direct
        metric = _OPCLASS_METRIC[opclass]
        vector_type = _OPCLASS_VECTOR_TYPE[opclass.split("_", 1)[0]]
    else:
        logging.debug("Ignoring vector index %s: unsupported opclass %s", row["index_name"], opclass)
        return None

    return VectorIndex(
        index_name=row["index_name"],
        table=table,
        collection_id=int(predicate.group(1)),
        strategy=strategy,
        dimensions=int(dimensions.group(1)),
        metric=metric,
        vector_type=vector_type,
    )


TABLES = ("embeddings_vector", "embeddings_halfvec")


async def discover_vector_indexes(conn: asyncpg.Connection) -> dict[int, list[VectorIndex]]:
    """
    Read every usable per-collection vector index, keyed by collection id.

    Args:
        conn (asyncpg.Connection): Any open connection; no special privileges are needed.

    Returns:
        dict[int, list[VectorIndex]]: Indexes per collection. A collection absent from the map
        has none, and search must fall back to the unindexed query.
    """
    rows = await conn.fetch(DISCOVERY_SQL, list(TABLES))

    by_collection: dict[int, list[VectorIndex]] = {}
    for row in rows:
        found = _parse(row["table_name"], row)
        if found is None:
            continue
        by_collection.setdefault(found.collection_id, []).append(found)

    logging.info(
        "Discovered %d usable vector index(es) across %d collection(s)",
        sum(len(v) for v in by_collection.values()),
        len(by_collection),
    )
    return by_collection


def choose_index(
    indexes: list[VectorIndex] | None,
    metric: DistanceMetric,
    dimensions: int,
) -> VectorIndex | None:
    """
    Pick the index search should target for this metric, or None to emit the unindexed query.

    A direct index is preferred when one matches: it needs no rescore step and loses no
    precision. A binary index is the fallback, and it can serve any metric because it only
    selects candidates -- the rescore ranks them by the requested metric exactly.

    `cosine_similarity` is served by a cosine *distance* index: search orders by the bare
    distance ascending and reports `1 - distance`, so the two share an ordering.

    Args:
        indexes (list[VectorIndex] | None): Candidates for this collection.
        metric (DistanceMetric): Metric the caller asked for.
        dimensions (int): Dimensionality the collection declares.

    Returns:
        VectorIndex | None: The index to target, or None if none can serve this query.
    """
    if not indexes:
        return None

    wanted = DistanceMetric.cosine_distance if metric == DistanceMetric.cosine_similarity else metric

    usable = [index for index in indexes if index.dimensions == dimensions]
    direct = [index for index in usable if index.strategy is IndexStrategy.direct and index.metric == wanted]
    if direct:
        return direct[0]

    binary = [index for index in usable if index.strategy is IndexStrategy.binary]
    return binary[0] if binary else None
