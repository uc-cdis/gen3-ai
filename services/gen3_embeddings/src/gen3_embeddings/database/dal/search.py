"""Data access methods for nearest-neighbour search over embeddings."""

from typing import Any

import asyncpg

from gen3_embeddings import config
from gen3_embeddings.database.dal.base import DataAccessLayerBase
from gen3_embeddings.database.helpers import (
    CollectionSearchTarget,
    build_multi_collection_search_sql,
    build_search_sql,
    get_embeddings_table_and_cast,
)
from gen3_embeddings.database.index_discovery import IndexStrategy, choose_index
from gen3_embeddings.database.models import Collection
from gen3_embeddings.models.schemas import DistanceMetric, VectorType


class SearchMixin(DataAccessLayerBase):
    """Vector similarity search, using whichever index each collection has."""

    def _binary_rescore_limit(
        self,
        collection_id: int,
        metric: DistanceMetric,
        dimensions: int,
        top_k: int,
    ) -> int | None:
        """
        How many candidates to rescore, or None if this search should not use a binary index.

        The pool has to be wider than `top_k`, because binary quantization keeps only the sign
        of each dimension and Hamming distance over that is a rough proxy for the real metric:
        a true nearest neighbour can sit well down the Hamming ordering, and only a candidate
        that was retrieved can be recovered by the exact rescore.

        Args:
            collection_id (int): Collection being searched.
            metric (DistanceMetric): Metric the caller asked for.
            dimensions (int): Dimensionality the collection declares.
            top_k (int): Rows the caller wants back.

        Returns:
            int | None: The candidate pool size, or None to emit the direct form.
        """
        chosen = choose_index(self.vector_indexes.get(collection_id), metric, dimensions)
        if chosen is None or chosen.strategy is not IndexStrategy.binary:
            return None

        wanted = top_k * config.BINARY_RESCORE_MULTIPLIER
        return max(min(wanted, config.BINARY_RESCORE_MAX), config.BINARY_RESCORE_MIN, top_k)

    async def search_embeddings_in_collection(
        self,
        collection: Collection,
        query_vector: list[float],
        top_k: int,
        min_value: float | None,
        max_value: float | None,
        distance_metric: DistanceMetric,
        filters: dict[str, str] | None,
    ) -> list[asyncpg.Record]:
        """
        Search embeddings in a collection for nearest neighbors of a query vector.

        Args:
            collection (Collection): Collection to search; its `vector_type` selects the table.
            query_vector (list[float]): Query vector to search against.
            top_k (int): Maximum number of results to return.
            min_value (float | None): Minimum similarity/distance threshold; rows outside it
                are excluded. Interpretation depends on `distance_metric`.
            max_value (float | None): Maximum similarity/distance threshold.
            distance_metric (DistanceMetric): Distance function to use, e.g. cosine or L2.
            filters (dict[str, str] | None): Metadata key/value filters; only rows matching
                all entries are considered.

        Returns:
            list[asyncpg.Record]: Matching rows including distance scores, ordered by
            distance. Visible to this caller under RLS. May be fewer than `top_k`.
        """
        vector_type = VectorType(collection.vector_type)
        table, _ = get_embeddings_table_and_cast(vector_type)
        rescore_limit = self._binary_rescore_limit(collection.id, distance_metric, collection.dimensions, top_k)

        # $1: collection_id, $2: vector, $3: top_k
        params: list[Any] = [collection.id, query_vector, top_k]

        sql, extra_params = build_search_sql(
            table=table,
            distance_metric=distance_metric,
            collection_id_param="$1::bigint",
            vector_placeholder="$2",
            top_k_param="$3::int",
            filters=filters,
            min_value=min_value,
            max_value=max_value,
            vector_type=vector_type,
            dimensions=collection.dimensions,
            binary_rescore_limit=rescore_limit,
        )
        params.extend(extra_params)

        async def _query(conn):
            stmt = await conn.prepare(sql)
            rows = await stmt.fetch(*params)
            return rows

        return await self._with_rls(_query, ef_search=rescore_limit)

    async def search_embeddings_across_collections(
        self,
        collections: list[Collection],
        query_vector: list[float],
        top_k: int,
        min_value: float | None,
        max_value: float | None,
        distance_metric: DistanceMetric,
        filters: dict[str, str] | None,
        vector_type: VectorType = VectorType.vector,
    ) -> list[asyncpg.Record]:
        """
        Search embeddings across multiple collections of the SAME vector_type.

        The collections list will be filtered to only those whose vector_type matches
        the given `vector_type` AND whose dimensions match the query vector length.
        A collection that matches neither cannot hold a hit for this query, so no
        collection matching means no hits: the result is empty rather than an error.

        Each surviving collection becomes one arm of a `UNION ALL`, carrying whichever query
        shape its own index can serve. See `build_multi_collection_search_sql` for why the
        older single-scan form could not use any per-collection index.

        Returns:
            list[asyncpg.Record]: The matching rows, at most `top_k` of them.
        """
        if not collections:
            return []

        # Filter collections by vector_type and dimensions
        filtered_collections: list[Collection] = []
        query_dims = len(query_vector)

        for col in collections:
            if col.vector_type == vector_type and col.dimensions == query_dims:
                filtered_collections.append(col)

        if not filtered_collections:
            return []

        table, _ = get_embeddings_table_and_cast(vector_type)

        # Each collection is indexed on its own, so the strategy is resolved per collection
        # rather than once for the search.
        targets = [
            CollectionSearchTarget(
                collection_id=col.id,
                binary_rescore_limit=self._binary_rescore_limit(col.id, distance_metric, query_dims, top_k),
            )
            for col in filtered_collections
        ]

        # $1: vector, $2: top_k. The collection ids are literals in the SQL, not parameters.
        params: list[Any] = [query_vector, top_k]

        sql, extra_params = build_multi_collection_search_sql(
            table=table,
            distance_metric=distance_metric,
            targets=targets,
            vector_placeholder="$1",
            top_k_param="$2::int",
            filters=filters,
            min_value=min_value,
            max_value=max_value,
            vector_type=vector_type,
            # every surviving collection matched the query's length, so they share one cast
            dimensions=query_dims,
        )
        params.extend(extra_params)

        async def _query(conn):
            stmt = await conn.prepare(sql)
            rows = await stmt.fetch(*params)
            return rows

        # One setting covers every arm, so it has to clear the widest pool any of them asks
        # for; a smaller value would silently truncate that arm's candidates.
        rescore_limits = [t.binary_rescore_limit for t in targets if t.binary_rescore_limit is not None]
        return await self._with_rls(_query, ef_search=max(rescore_limits, default=None))
