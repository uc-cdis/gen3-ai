"""SQL construction helpers for the embeddings tables."""

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


def build_search_sql(
    table: str,
    distance_metric: DistanceMetric,
    single_collection: bool,
    # "$1" or "$1::bigint[]"
    collection_ids_param: str,
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
    Build the SQL and parameter list for search queries on the embeddings_* tables.

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

    The ordering expression is also kept separate from the reported value. `cosine_similarity`
    reports `1 - distance`, and ordering by that arithmetic wrapper is not something an index
    can serve; ordering by the bare distance ascending produces the identical ranking and is
    index-eligible. `min_value`/`max_value` still compare against the reported value, so the
    two expressions cannot be collapsed into one.

    Args:
        table (str): Target table, `embeddings_vector` or `embeddings_halfvec`.
        distance_metric (DistanceMetric): Metric to rank by.
        single_collection (bool): Whether the collection filter is a scalar or an array.
        collection_ids_param (str): Placeholder for the collection filter.
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
    filters = filters or {}
    params: list[Any] = []
    where_clauses: list[str] = []

    # 1) collection filter
    if single_collection:
        # "$1"
        where_clauses.append("collection_id = " + collection_ids_param)
    else:
        # "$1::bigint[]"
        where_clauses.append("collection_id = ANY(" + collection_ids_param + ")")

    # 2) metric expressions. `order_expr` is what an index can serve; `value_expr` is what
    #    the caller sees and what the thresholds compare against.
    try:
        operator = _METRIC_OPERATORS[distance_metric]
    except KeyError:
        raise ValueError(f"Unsupported distance metric: {distance_metric}") from None

    # `vector_type` may arrive as the plain string rather than the enum -- it is a StrEnum so
    # callers compare it either way, and `search_embeddings_across_collections` passes through
    # whatever it was given. Coerce before reaching for `.value`.
    typed = f"{VectorType(vector_type).value}({dimensions})"
    order_expr = f"embedding::{typed} {operator} {vector_placeholder}::{typed}"
    value_expr = f"1 - ({order_expr})" if distance_metric == DistanceMetric.cosine_similarity else order_expr

    # 3) filters on metadata (after the first 3 parameters)
    param_index = 4
    for k, v in filters.items():
        where_clauses.append(f"metadata->>(${param_index}::text) = ${param_index + 1}::text")
        params.extend((k, v))
        param_index += 2

    # 4) min/max constraints on the reported value, which go *outside* a materialized CTE
    #    rather than into the WHERE above.
    #
    #    A filter on the ordered expression sits below the LIMIT, and the executor cannot end
    #    an ordered index scan early while rows are still being filtered out: it walks the
    #    whole index instead, reporting the remainder as "Rows Removed by Filter". That is the
    #    full-scan cost this rewrite exists to avoid, so the threshold has to be applied after
    #    the limit. The pgvector README prescribes exactly this shape ("use a materialized CTE
    #    and place the distance filter outside of it"); the underlying executor behaviour is
    #    https://www.postgresql.org/message-id/flat/CAOdR5yGUoMQ6j7M5hNUXrySzaqZVGf_Ne%2B8fwZMRKTFxU1nbJg%40mail.gmail.com
    #
    #    Every other filter stays inside the CTE, where `hnsw.iterative_scan` can keep feeding
    #    the scan until `top_k` rows survive them.
    #
    #    The tradeoff is that the limit now applies before the thresholds. For the bound that
    #    drops *distant* hits -- `max_value` on a distance metric, `min_value` on
    #    cosine_similarity -- that changes nothing, because everything it removes sorts after
    #    everything it keeps. The opposite bound drops the *nearest* hits, so it eats into the
    #    rows the CTE already limited to `top_k` and such a search can return fewer than
    #    `top_k` hits even when more would have qualified.
    threshold_clauses: list[str] = []
    if min_value is not None:
        threshold_clauses.append(f"value >= ${param_index}")
        params.append(min_value)
        param_index += 1

    if max_value is not None:
        threshold_clauses.append(f"value <= ${param_index}")
        params.append(max_value)
        param_index += 1

    where_sql = " AND ".join(where_clauses)
    threshold_sql = f"WHERE {' AND '.join(threshold_clauses)}" if threshold_clauses else ""

    # cosine_similarity reports `1 - distance`, whose values descend as the distance ascends,
    # so re-sorting on the reported value has to flip direction for it.
    value_order = "DESC" if distance_metric == DistanceMetric.cosine_similarity else "ASC"

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
        #
        #     Only the left operand carries the `::bit(n)` cast, matching the indexed
        #     expression exactly; `binary_quantize()` on the query vector already yields a bit
        #     string of the same length.
        bits = f"bit({dimensions})"
        hamming_expr = f"binary_quantize(embedding)::{bits} <~> binary_quantize({vector_placeholder}::{typed})"

        candidates_sql = f"""
            SELECT *,
                   {value_expr} AS value
            FROM {table}
            WHERE {where_sql}
            ORDER BY {hamming_expr} ASC
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
