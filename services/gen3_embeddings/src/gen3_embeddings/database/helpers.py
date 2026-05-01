from typing import Any

from gen3_embeddings.models.schemas import DistanceMetric, VectorType


def get_embeddings_table_and_cast(vector_type: VectorType) -> tuple[str, str]:
    """
    Return (table_name, sql_cast) given a vector type.

    Example:
      VectorType.vector   -> ("embeddings_vector", "::vector")
      VectorType.halfvec  -> ("embeddings_halfvec", "::halfvec")
    """
    if vector_type == VectorType.vector:
        return "embeddings_vector", "::vector"
    if vector_type == VectorType.halfvec:
        return "embeddings_halfvec", "::halfvec"
    raise ValueError(f"Unsupported vector type: {vector_type}")


def build_search_sql(
    table: str,
    distance_metric: DistanceMetric,
    single_collection: bool,
    collection_ids_param: str,  # "$1" or "$1::bigint[]"
    vector_param: str,  # e.g. "$2::vector" or "$2::halfvec"
    top_k_param: str,  # "$3"
    filters: dict[str, str] | None,
    min_value: float | None,
    max_value: float | None,
) -> tuple[str, list[Any]]:
    """
    Build the SQL and parameter list for search queries on the embeddings_* tables.
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

    # 2) metric expression and order
    col_name = "embedding"

    if distance_metric == DistanceMetric.l2_distance:
        expr = f"{col_name} <-> {vector_param}"
        ascending = True
    elif distance_metric == DistanceMetric.inner_product:
        expr = f"{col_name} <#> {vector_param}"
        ascending = True
    elif distance_metric == DistanceMetric.cosine_distance:
        expr = f"{col_name} <=> {vector_param}"
        ascending = True
    elif distance_metric == DistanceMetric.l1_distance:
        expr = f"{col_name} <+> {vector_param}"
        ascending = True
    elif distance_metric == DistanceMetric.cosine_similarity:
        expr = f"1 - ({col_name} <=> {vector_param})"
        ascending = False
    else:
        raise ValueError(f"Unsupported distance metric: {distance_metric}")

    # 3) filters on metadata (after the first 3 parameters)
    param_index = 4
    for k, v in filters.items():
        where_clauses.append(f"metadata->>$${k}$$ = ${param_index}")
        params.append(v)
        param_index += 1

    # 4) min/max constraints on the expression
    if min_value is not None:
        where_clauses.append(f"{expr} >= ${param_index}")
        params.append(min_value)
        param_index += 1

    if max_value is not None:
        where_clauses.append(f"{expr} <= ${param_index}")
        params.append(max_value)
        param_index += 1

    where_sql = " AND ".join(where_clauses)
    order_dir = "ASC" if ascending else "DESC"

    sql = f"""
        SELECT *,
               {expr} AS value
        FROM {table}
        WHERE {where_sql}
        ORDER BY {expr} {order_dir}
        LIMIT {top_k_param}
    """

    return sql, params
