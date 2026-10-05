"""SQL construction helpers for the embeddings tables."""

from dataclasses import dataclass
from typing import Any

from gen3_embeddings.models.schemas import DistanceMetric, VectorType


def affected_row_count(command_tag: str) -> int:
    """
    Return how many rows a statement affected, from its Postgres command tag.

    `Connection.execute` returns the command tag rather than a row count, e.g. "DELETE 3",
    "DELETE 0" or "UPDATE 2". Only the trailing count says whether anything changed: a
    statement that matched nothing still reports its verb, so testing the verb alone (e.g.
    `tag.startswith("DELETE")`) is always true and reads "nothing matched" as success.

    Args:
        command_tag (str): Command tag as returned by `Connection.execute`.

    Returns:
        int: Number of rows affected, or 0 if the tag carries no parsable count.
    """
    try:
        return int(command_tag.rsplit(" ", 1)[-1])
    except (ValueError, IndexError):
        # no trailing count means nothing was reported as affected
        return 0


def get_embeddings_table_and_cast(vector_type: VectorType) -> tuple[str, str]:
    """
    Return (table_name, sql_cast) given a vector type.

    Example:
      VectorType.vector   -> ("embeddings_vector", "::vector")
      VectorType.halfvec  -> ("embeddings_halfvec", "::halfvec")

    Args:
        vector_type (VectorType): Storage type of the collection.

    Returns:
        tuple[str, str]: The table holding that type, and the pgvector cast for its column.

    Raises:
        ValueError: If the type has no table.
    """
    if vector_type == VectorType.vector:
        return "embeddings_vector", "::vector"
    if vector_type == VectorType.halfvec:
        return "embeddings_halfvec", "::halfvec"
    raise ValueError(f"Unsupported vector type: {vector_type}")


# pgvector's ordering operators. `inner_product` maps to `<#>`, which is the *negative*
# inner product, so every one of these is "smaller is closer" and every search orders
# ascending. `cosine_similarity` shares `<=>` with `cosine_distance` and differs only in
# how the value is reported, which is the whole point of separating value from order below.
_METRIC_OPERATORS: dict[DistanceMetric, str] = {
    DistanceMetric.l2_distance: "<->",
    DistanceMetric.inner_product: "<#>",
    DistanceMetric.cosine_distance: "<=>",
    DistanceMetric.l1_distance: "<+>",
    DistanceMetric.cosine_similarity: "<=>",
}


@dataclass(frozen=True)
class MetricSql:
    """The four SQL fragments a metric contributes to a search."""

    # What an index can serve. Always ascending: every pgvector operator is "smaller is closer".
    order: str
    # What the caller is shown, and what `min_value`/`max_value` compare against.
    value: str
    # Hamming ordering over `binary_quantize()`, for collections indexed that way.
    hamming: str
    # Direction to re-sort the reported value in, once ordering is no longer the index's job.
    value_order: str


def _metric_sql(
    distance_metric: DistanceMetric,
    vector_type: VectorType,
    dimensions: int,
    vector_placeholder: str,
) -> MetricSql:
    """
    Build the metric expressions shared by every shape of search query.

    The ordering expression is kept separate from the reported value. `cosine_similarity`
    reports `1 - distance`, and ordering by that arithmetic wrapper is not something an index
    can serve; ordering by the bare distance ascending produces the identical ranking and is
    index-eligible. `min_value`/`max_value` still compare against the reported value, so the
    two expressions cannot be collapsed into one.

    Args:
        distance_metric (DistanceMetric): Metric to rank by.
        vector_type (VectorType): Storage type of the collection being searched.
        dimensions (int): Dimensionality the collection declares, used in the cast.
        vector_placeholder (str): Bare placeholder for the query vector, e.g. "$2".

    Returns:
        MetricSql: The ordering, value, Hamming, and re-sort direction fragments.

    Raises:
        ValueError: If the metric has no pgvector operator.
    """
    try:
        operator = _METRIC_OPERATORS[distance_metric]
    except KeyError:
        raise ValueError(f"Unsupported distance metric: {distance_metric}") from None

    # `vector_type` may arrive as the plain string rather than the enum -- it is a StrEnum so
    # callers compare it either way, and the cross-collection search passes through whatever
    # it was given. Coerce before reaching for `.value`.
    typed = f"{VectorType(vector_type).value}({dimensions})"
    order = f"embedding::{typed} {operator} {vector_placeholder}::{typed}"

    # Only the indexed expression (left operand) is explicitly cast to `bit(n)`.
    # The query operand relies on `binary_quantize()` to produce a compatible bit
    # string length, so no explicit `::bit(n)` cast is applied on the right.
    hamming = f"binary_quantize(embedding)::bit({dimensions}) <~> binary_quantize({vector_placeholder}::{typed})"

    is_similarity = distance_metric == DistanceMetric.cosine_similarity
    return MetricSql(
        order=order,
        value=f"1 - ({order})" if is_similarity else order,
        hamming=hamming,
        # cosine_similarity reports `1 - distance`, whose values descend as the distance
        # ascends, so re-sorting on the reported value has to flip direction for it.
        value_order="DESC" if is_similarity else "ASC",
    )


def _metadata_filter_clauses(
    filters: dict[str, str] | None,
    param_index: int,
) -> tuple[list[str], list[Any], int]:
    """
    Build the metadata equality clauses, which stay inside the candidate query.

    Args:
        filters (dict[str, str] | None): Metadata equality filters.
        param_index (int): Number of the next free placeholder.

    Returns:
        tuple[list[str], list[Any], int]: Clauses, their parameters, and the next free
        placeholder number.
    """
    clauses: list[str] = []
    params: list[Any] = []
    for key, value in (filters or {}).items():
        clauses.append(f"metadata->>(${param_index}::text) = ${param_index + 1}::text")
        params.extend((key, value))
        param_index += 2
    return clauses, params, param_index


def _threshold_clauses(
    min_value: float | None,
    max_value: float | None,
    param_index: int,
) -> tuple[list[str], list[Any], int]:
    """
    Build the `min_value`/`max_value` clauses, which go *outside* the candidate query.

    A filter on the ordered expression sits below the LIMIT, and the executor cannot end an
    ordered index scan early while rows are still being filtered out: it walks the whole index
    instead, reporting the remainder as "Rows Removed by Filter". That is a full-scan cost,
    so the threshold has to be applied after the limit. The
    pgvector README prescribes exactly this shape ("use a materialized CTE and place the
    distance filter outside of it"); the underlying executor behaviour is
    https://www.postgresql.org/message-id/flat/CAOdR5yGUoMQ6j7M5hNUXrySzaqZVGf_Ne%2B8fwZMRKTFxU1nbJg%40mail.gmail.com

    Every other filter stays inside, where `hnsw.iterative_scan` can keep feeding the scan
    until `top_k` rows survive them.

    The tradeoff is that the limit applies before the thresholds. For the bound that drops
    *distant* hits -- `max_value` on a distance metric, `min_value` on cosine_similarity --
    that changes nothing, because everything it removes sorts after everything it keeps. The
    opposite bound drops the *nearest* hits, so it eats into the rows already limited to
    `top_k`, and such a search can return fewer than `top_k` hits even when more would have
    qualified.

    Args:
        min_value (float | None): Lower bound on the reported value.
        max_value (float | None): Upper bound on the reported value.
        param_index (int): Number of the next free placeholder.

    Returns:
        tuple[list[str], list[Any], int]: Clauses, their parameters, and the next free
        placeholder number.
    """
    clauses: list[str] = []
    params: list[Any] = []
    if min_value is not None:
        clauses.append(f"value >= ${param_index}")
        params.append(min_value)
        param_index += 1
    if max_value is not None:
        clauses.append(f"value <= ${param_index}")
        params.append(max_value)
        param_index += 1
    return clauses, params, param_index


def build_search_sql(
    table: str,
    distance_metric: DistanceMetric,
    # "$1::bigint"
    collection_id_param: str,
    # bare placeholder for the query vector, e.g. "$2"; this function adds the cast, because
    # the cast has to match the index expression exactly and that is decided here
    vector_placeholder: str,
    # "$3"
    top_k_param: str,
    filters: dict[str, str] | None,
    min_value: float | None,
    max_value: float | None,
    vector_type: VectorType,
    dimensions: int,
    binary_rescore_limit: int | None = None,
) -> tuple[str, list[Any]]:
    """
    Build the SQL and parameter list for searching a single collection.

    Two details decide whether this query can use an index at all.

    The `embedding` columns are declared without a dimension (`VECTOR`, not `VECTOR(n)`),
    because one table holds every collection and collections differ in dimensionality.
    pgvector cannot build HNSW or IVFFlat on a dimensionless column, so the only indexable
    form is a partial expression index per collection:

        CREATE INDEX ... USING hnsw ((embedding::vector(256)) vector_cosine_ops)
            WHERE collection_id = 1;

    Postgres only uses an expression index when the query's expression matches the indexed
    one exactly, so the `::vector(n)` cast written here is load-bearing: drop it, or emit a
    different dimension, and the plan silently falls back to a sequential scan.

    See `_metric_sql` for why ordering and the reported value are separate expressions, and
    `build_multi_collection_search_sql` for the cross-collection shape.

    Args:
        table (str): Target table, `embeddings_vector` or `embeddings_halfvec`.
        distance_metric (DistanceMetric): Metric to rank by.
        collection_id_param (str): Placeholder for the collection filter, e.g. "$1::bigint".
        vector_placeholder (str): Bare placeholder for the query vector, e.g. "$2".
        top_k_param (str): Placeholder for the row limit.
        filters (dict[str, str] | None): Metadata equality filters.
        min_value (float | None): Lower bound on the reported value.
        max_value (float | None): Upper bound on the reported value.
        vector_type (VectorType): Storage type of the collection being searched.
        dimensions (int): Dimensionality the collection declares, used in the cast.
        binary_rescore_limit (int | None): How many candidates to pull from a binary-quantized
            index before re-ranking them exactly. None emits the direct form instead. Must be
            >= `top_k`, and the caller must raise `hnsw.ef_search` to at least this value or
            the index returns fewer candidates than asked for, silently.

    Returns:
        tuple[str, list[Any]]: The SQL, and the parameters that follow the first three.

    Raises:
        ValueError: If the metric has no pgvector operator.
    """
    metric = _metric_sql(distance_metric, vector_type, dimensions, vector_placeholder)
    order_expr, value_expr, value_order = metric.order, metric.value, metric.value_order

    # The first three placeholders are the collection, the vector, and the limit, so anything
    # this function adds starts at $4.
    filter_clauses, params, param_index = _metadata_filter_clauses(filters, 4)
    threshold_clauses, threshold_params, _ = _threshold_clauses(min_value, max_value, param_index)
    params.extend(threshold_params)

    where_sql = " AND ".join([f"collection_id = {collection_id_param}", *filter_clauses])
    threshold_sql = f"WHERE {' AND '.join(threshold_clauses)}" if threshold_clauses else ""

    if binary_rescore_limit is not None:
        # 5a) Binary-quantized retrieval, for collections whose index is over
        #     `binary_quantize(embedding)::bit(n)`.
        #
        #     Two stages. The inner query picks candidates by Hamming distance on the bit
        #     index, which is cheap -- a bit(1536) is 200 bytes against ~6KB for the fp32
        #     vector, so ~40 candidates share a page instead of one straddling several. The
        #     outer query then re-ranks those candidates by the exact metric, which is what
        #     recovers the precision quantization threw away.
        #
        #     Because the Hamming ordering only decides WHICH rows to look at and never what
        #     the caller is shown, one such index serves every metric.
        candidates_sql = f"""
            SELECT *,
                   {value_expr} AS value
            FROM {table}
            WHERE {where_sql}
            ORDER BY {metric.hamming} ASC
            LIMIT {binary_rescore_limit}
        """
        sql = f"""
            WITH candidates AS MATERIALIZED ({candidates_sql})
            SELECT *
            FROM candidates
            {threshold_sql}
            ORDER BY value {value_order}
            LIMIT {top_k_param}
        """
        return sql, params

    # 5b) Direct retrieval: the ordering expression is the one the index was built on.
    nearest_sql = f"""
        SELECT *,
               {value_expr} AS value
        FROM {table}
        WHERE {where_sql}
        ORDER BY {order_expr} ASC
        LIMIT {top_k_param}
    """

    if not threshold_clauses:
        return nearest_sql, params

    # The CTE already ordered and limited, so the outer sort only reorders at most `top_k` rows.
    sql = f"""
        WITH nearest AS MATERIALIZED ({nearest_sql})
        SELECT *
        FROM nearest
        {threshold_sql}
        ORDER BY value {value_order}
    """

    return sql, params


@dataclass(frozen=True)
class CollectionSearchTarget:
    """
    One collection a cross-collection search covers, and how its index wants to be queried.

    `binary_rescore_limit` is per collection rather than per search because collections are
    indexed independently: one may have an fp32 index, the next only a binary-quantized one,
    and the next none at all. Each gets the query shape its own index can serve.
    """

    collection_id: int
    # Candidate pool for a binary-quantized index; None means emit the direct form, which is
    # also what an unindexed collection gets.
    binary_rescore_limit: int | None = None


def build_multi_collection_search_sql(
    table: str,
    distance_metric: DistanceMetric,
    targets: list[CollectionSearchTarget],
    # bare placeholder for the query vector, e.g. "$1"
    vector_placeholder: str,
    # "$2::int"
    top_k_param: str,
    filters: dict[str, str] | None,
    min_value: float | None,
    max_value: float | None,
    vector_type: VectorType,
    dimensions: int,
    first_param_index: int = 3,
) -> tuple[str, list[Any]]:
    """
    Build the SQL and parameter list for searching several collections at once.

    This is a `UNION ALL` of one arm per collection rather than a single scan filtered by
    `collection_id = ANY($1)`, for two reasons.

    The indexes are **partial**, one per collection, with predicates like
    `WHERE collection_id = 101`. Postgres uses a partial index only when it can prove the
    query implies that predicate, and `collection_id = ANY($1::bigint[])` proves nothing about
    any single collection, so the array form cannot use a single one of them -- it is a
    sequential scan over every collection in the table no matter how many are indexed. Each
    arm here restricts to **one** collection id, written as a literal so the implication holds
    in a generic plan as well as a custom one, which is what makes each arm index-eligible.

    The arms also differ in shape, because `choose_index` may land on a direct index for one
    collection and a binary-quantized one for the next. A single query body can only have one
    ordering expression, so a mixed set has to be split regardless of the predicate question.

    Each arm takes its own `top_k`, and the outer query re-sorts and takes `top_k` again. That
    is exactly equivalent to a global top-k: no collection can contribute more than `top_k`
    rows to the final answer, so nothing that would have survived is discarded early.

    Args:
        table (str): Target table, `embeddings_vector` or `embeddings_halfvec`.
        distance_metric (DistanceMetric): Metric to rank by.
        targets (list[CollectionSearchTarget]): Collections to search, with their index shapes.
            All must share `dimensions` and `vector_type`; the caller filters out any that do
            not, since one query body carries one cast.
        vector_placeholder (str): Bare placeholder for the query vector, e.g. "$1".
        top_k_param (str): Placeholder for the row limit, e.g. "$2::int".
        filters (dict[str, str] | None): Metadata equality filters, applied inside every arm.
        min_value (float | None): Lower bound on the reported value.
        max_value (float | None): Upper bound on the reported value.
        vector_type (VectorType): Storage type shared by the collections.
        dimensions (int): Dimensionality shared by the collections, used in the cast.
        first_param_index (int): Number of the first placeholder this function may allocate.

    Returns:
        tuple[str, list[Any]]: The SQL, and the parameters that follow the caller's own.

    Raises:
        ValueError: If `targets` is empty, or the metric has no pgvector operator.
    """
    if not targets:
        raise ValueError("build_multi_collection_search_sql needs at least one collection to search")

    metric = _metric_sql(distance_metric, vector_type, dimensions, vector_placeholder)

    filter_clauses, params, param_index = _metadata_filter_clauses(filters, first_param_index)
    threshold_clauses, threshold_params, _ = _threshold_clauses(min_value, max_value, param_index)
    params.extend(threshold_params)

    # Every arm shares these placeholders; the filters are evaluated once per arm but bind the
    # same parameters, so the union adds no parameters of its own.
    filter_sql = "".join(f" AND {clause}" for clause in filter_clauses)

    arms: list[str] = []
    for target in targets:
        # A literal, not a placeholder, so the partial index predicate is provable. `int()` is
        # what keeps that safe: these ids come from `collections.id`, and anything that is not
        # an integer raises here rather than reaching the SQL text.
        collection_id = int(target.collection_id)
        candidates = f"SELECT *, {metric.value} AS value FROM {table} WHERE collection_id = {collection_id}{filter_sql}"

        if target.binary_rescore_limit is None:
            arms.append(f"({candidates} ORDER BY {metric.order} ASC LIMIT {top_k_param})")
        else:
            # Same two-stage shape as the single-collection binary path: Hamming picks the
            # candidates, the exact metric ranks them. The inner LIMIT stops the subquery from
            # being flattened, so the re-rank really does run on the retrieved pool.
            arms.append(
                f"(SELECT * FROM ({candidates}"
                f" ORDER BY {metric.hamming} ASC LIMIT {int(target.binary_rescore_limit)})"
                f" AS rescored_{collection_id} ORDER BY value {metric.value_order} LIMIT {top_k_param})"
            )

    union_sql = "\n            UNION ALL\n            ".join(arms)
    threshold_sql = f"WHERE {' AND '.join(threshold_clauses)}" if threshold_clauses else ""

    # The arms always go through a CTE, even with nothing to filter and only one of them.
    #
    # A trailing `ORDER BY` after a set operation belongs to the union, but with a single arm
    # there is no set operation and it lands on a SELECT that already carries one, which
    # Postgres rejects outright ("multiple ORDER BY clauses not allowed"). Wrapping removes
    # that special case, and MATERIALIZED is needed anyway the moment a threshold appears.
    #
    # It costs nothing: the arms are individually ordered and limited, so at most
    # len(targets) * top_k rows are materialized, and they all have to be sorted regardless.
    return (
        f"""
        WITH candidates AS MATERIALIZED (
            {union_sql}
        )
        SELECT *
        FROM candidates
        {threshold_sql}
        ORDER BY value {metric.value_order}
        LIMIT {top_k_param}
    """,
        params,
    )
