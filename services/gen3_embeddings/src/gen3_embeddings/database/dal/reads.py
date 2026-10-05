"""Data access methods that read collections and embeddings."""

from uuid import UUID

from gen3_embeddings.database.dal.base import DataAccessLayerBase
from gen3_embeddings.database.errors import InvalidCollectionNameError
from gen3_embeddings.database.helpers import get_embeddings_table_and_cast
from gen3_embeddings.database.models import Collection, Embedding
from gen3_embeddings.models.helpers import normalize_collection_name
from gen3_embeddings.models.schemas import VectorType


class ReadMixin(DataAccessLayerBase):
    """Lookups, listings and counts. Every result is filtered by RLS to what the caller may see."""

    async def get_collection_by_name(self, collection_name: str) -> Collection | None:
        """
        Look up a collection by name, if the caller is allowed to see it.

        Args:
            collection_name (str): Name of the collection; normalized before lookup.

        Returns:
            Collection | None: The collection, or None if it does not exist **or** RLS hid
            it from this caller. The two cases are deliberately indistinguishable so callers
            cannot probe for collection names; callers typically surface this as a 404.

        Raises:
            InvalidCollectionNameError: If `collection_name` is not a valid collection name.
            asyncpg.InsufficientPrivilegeError: If the database role is missing a GRANT the
                query needs. A deployment fault rather than the caller's, so it stays a 500.
            asyncpg.QueryCanceledError: If the query runs past `DB_STATEMENT_TIMEOUT_MS`.
            TypeError: If a `collections` row has a column `Collection` does not mirror, as a
                migration that adds one ahead of the code would cause.
            ValueError: If a stored `vector_type` is not a known `VectorType`.
        """
        try:
            collection_name = normalize_collection_name(collection_name)
        except ValueError as exc:
            raise InvalidCollectionNameError(str(exc)) from exc

        async def _query(conn):
            stmt = await conn.prepare("SELECT * FROM collections WHERE collection_name = $1::text")
            row = await stmt.fetchrow(collection_name)
            return Collection.from_record(row) if row else None

        return await self._with_rls(_query)

    async def get_collection_by_id(self, collection_id: int) -> Collection | None:
        """
        Look up a collection by primary key, if the caller is allowed to see it.

        Args:
            collection_id (int): Primary key of the collection.

        Returns:
            Collection | None: The collection, or None if it does not exist **or** RLS hid
            it from this caller.

        Raises:
            asyncpg.InsufficientPrivilegeError: If the database role is missing a GRANT the
                query needs. A deployment fault rather than the caller's, so it stays a 500.
            asyncpg.QueryCanceledError: If the query runs past `DB_STATEMENT_TIMEOUT_MS`.
            TypeError: If a `collections` row has a column `Collection` does not mirror, as a
                migration that adds one ahead of the code would cause.
            ValueError: If a stored `vector_type` is not a known `VectorType`.
        """

        async def _query(conn):
            stmt = await conn.prepare("SELECT * FROM collections WHERE id = $1::bigint")
            row = await stmt.fetchrow(collection_id)
            return Collection.from_record(row) if row else None

        return await self._with_rls(_query)

    async def list_collections(
        self,
        offset: int = 0,
        limit: int = 100,
    ) -> list[Collection]:
        """
        List the collections the caller is authorized for.

        This uses the table's RLS policy, so it lists only the collections the caller is
        authorized for, and the SQL is a plain paged SELECT.

        Args:
            offset (int): Number of rows to skip.
            limit (int): Maximum number of rows to return. Callers that need every
                authorized collection must page; the default silently caps at 100.

        Returns:
            list[Collection]: Authorized collections for this page, empty if the caller
            has no allowed collections. Never more rows than `allowed_collection_names`
            has entries, since that set is the whole candidate space - so a caller that
            wants every collection and needs to know whether it got them all can ask for
            one more than its own ceiling and check whether that extra row came back.

        Raises:
            asyncpg.InsufficientPrivilegeError: If the database role is missing a GRANT the
                query needs. A deployment fault rather than the caller's, so it stays a 500.
            asyncpg.QueryCanceledError: If the query runs past `DB_STATEMENT_TIMEOUT_MS`.
            TypeError: If a `collections` row has a column `Collection` does not mirror, as a
                migration that adds one ahead of the code would cause.
            ValueError: If a stored `vector_type` is not a known `VectorType`.
        """
        # nothing can be visible, so skip the round trip
        if not self.allowed_collection_names:
            return []

        async def _query(conn):
            stmt = await conn.prepare(
                """
                SELECT *
                FROM collections
                ORDER BY collection_name
                LIMIT $2::int
                OFFSET $1::int
                """
            )
            rows = await stmt.fetch(offset, limit)
            return [Collection.from_record(r) for r in rows]

        return await self._with_rls(_query)

    async def get_embedding_by_collection_and_id(
        self,
        collection: Collection,
        embedding_id: UUID,
    ) -> Embedding | None:
        """
        Read a single embedding from a collection.

        Args:
            collection (Collection): Collection the embedding belongs to; its `vector_type`
                selects which embeddings table is queried.
            embedding_id (UUID): Identifier of the embedding.

        Returns:
            Embedding | None: The embedding, or None if it does not exist **or** RLS hid it
            from this caller.

        Raises:
            asyncpg.InsufficientPrivilegeError: If the database role is missing a GRANT the
                query needs. A deployment fault rather than the caller's, so it stays a 500.
            asyncpg.QueryCanceledError: If the query runs past `DB_STATEMENT_TIMEOUT_MS`.
        """
        table, _ = get_embeddings_table_and_cast(VectorType(collection.vector_type))

        async def _query(conn):
            stmt = await conn.prepare(
                f"SELECT * FROM {table} WHERE collection_id = $1::bigint AND embedding_id = $2::uuid"
            )
            row = await stmt.fetchrow(collection.id, embedding_id)
            return Embedding.from_record(row) if row else None

        return await self._with_rls(_query)

    async def list_embeddings_in_collection(
        self,
        collection: Collection,
        offset: int,
        limit: int,
    ) -> list[Embedding]:
        """
        List embeddings in a collection, oldest first.

        Args:
            collection (Collection): Collection to read from.
            offset (int): Number of rows to skip.
            limit (int): Maximum number of rows to return.

        Returns:
            list[Embedding]: Embeddings visible to this caller under RLS. Ordering is by
            `created_at`, which is not unique, so rows can shift between pages.

        Raises:
            asyncpg.InsufficientPrivilegeError: If the database role is missing a GRANT the
                query needs. A deployment fault rather than the caller's, so it stays a 500.
            asyncpg.QueryCanceledError: If the query runs past `DB_STATEMENT_TIMEOUT_MS`.
        """
        table, _ = get_embeddings_table_and_cast(VectorType(collection.vector_type))

        async def _query(conn):
            stmt = await conn.prepare(
                f"""
                SELECT * FROM {table}
                WHERE collection_id = $1::bigint
                ORDER BY embedding_id
                OFFSET $2::int
                LIMIT $3::int
                """
            )
            rows = await stmt.fetch(collection.id, offset, limit)
            return [Embedding.from_record(r) for r in rows]

        return await self._with_rls(_query)

    async def get_embeddings_bulk(
        self,
        embedding_ids: list[UUID],
        vector_type: VectorType | None,
        collection_id: int | None = None,
    ) -> list[Embedding]:
        """
        Fetch embeddings by ID from one or both embeddings tables.

        Args:
            embedding_ids (list[UUID]): Identifiers to fetch.
            vector_type (VectorType | None): If given, only that table is queried. If None,
                both `embeddings_vector` and `embeddings_halfvec` are queried and results
                combined.
            collection_id (int | None): If given, additionally filter by collection.

        Returns:
            list[Embedding]: Embeddings visible to this caller under RLS, in no guaranteed
            order. May be shorter than `embedding_ids` if any are hidden by RLS or do not
            exist.

        Raises:
            asyncpg.InsufficientPrivilegeError: If the database role is missing a GRANT the
                query needs. A deployment fault rather than the caller's, so it stays a 500.
            asyncpg.QueryCanceledError: If the query runs past `DB_STATEMENT_TIMEOUT_MS`.
        """

        async def _query(conn):
            results: list[Embedding] = []

            def rows_to_embeddings(rows):
                return [Embedding.from_record(r) for r in rows]

            if vector_type:
                table, _ = get_embeddings_table_and_cast(vector_type)
                raw_stmt = f"SELECT * FROM {table} WHERE embedding_id = ANY($1::uuid[])"

                if collection_id:
                    raw_stmt += f" AND collection_id = {collection_id}"

                stmt = await conn.prepare(raw_stmt)
                rows = await stmt.fetch(embedding_ids)
                results.extend(rows_to_embeddings(rows))
            else:
                # query both vector and halfvec tables
                for vt in (VectorType.vector, VectorType.halfvec):
                    table, _ = get_embeddings_table_and_cast(vt)
                    raw_stmt = f"SELECT * FROM {table} WHERE embedding_id = ANY($1::uuid[])"

                    if collection_id:
                        raw_stmt += f" AND collection_id = {collection_id}"

                    stmt = await conn.prepare(raw_stmt)

                    rows = await stmt.fetch(embedding_ids)
                    results.extend(rows_to_embeddings(rows))

            return results

        return await self._with_rls(_query)

    async def get_embeddings_bulk_from_collection_ordered(
        self,
        embedding_ids: list[UUID],
        collection: Collection,
    ) -> list[tuple[int, Embedding]]:
        """
        Fetch specific embeddings from a collection, tagged with their input position.

        The caller supplies an ordered list of ids and gets back the index each row had in
        that list, so results can be lined up with the request even though rows the caller
        cannot see are simply absent.

        Args:
            embedding_ids (list[UUID]): Embedding ids to fetch, in request order.
            collection (Collection): Collection to read from.

        Returns:
            list[tuple[int, Embedding]]: (input index, embedding) pairs in request order.
            Ids that do not exist or are hidden by RLS are omitted, so this may be shorter
            than `embedding_ids`.

        Raises:
            asyncpg.InsufficientPrivilegeError: If the database role is missing a GRANT the
                query needs. A deployment fault rather than the caller's, so it stays a 500.
            asyncpg.QueryCanceledError: If the query runs past `DB_STATEMENT_TIMEOUT_MS`.
        """
        if not embedding_ids:
            return []

        table, _ = get_embeddings_table_and_cast(VectorType(collection.vector_type))

        async def _query(conn):
            stmt = await conn.prepare(
                f"""
                SELECT
                    e.collection_id,
                    e.embedding_id,
                    e.embedding,
                    e.authz,
                    e.metadata,
                    e.created_at,
                    e.updated_at,
                    inp.ord
                FROM unnest($1::uuid[]) WITH ORDINALITY AS inp(embedding_id, ord)
                JOIN {table} e
                ON e.embedding_id = inp.embedding_id
                WHERE e.collection_id = $2::bigint
                ORDER BY inp.ord
                """
            )
            rows = await stmt.fetch(embedding_ids, collection.id)

            results: list[tuple[int, Embedding]] = []
            for row in rows:
                input_index = row["ord"] - 1
                emb = Embedding.from_record(row)
                results.append((input_index, emb))
            return results

        return await self._with_rls(_query)

    async def get_collection_by_id_bulk(self, collection_ids: list[int]) -> list[Collection]:
        """
        Fetch several collections by primary key, keeping only those the caller may see.

        This uses the table's RLS policy, so it filters out the ids the caller is not
        authorized for: those simply return no row.

        Args:
            collection_ids (list[int]): Primary keys to look up.

        Returns:
            list[Collection]: Authorized collections, in no guaranteed order. May be
            shorter than `collection_ids`, and empty if the caller has no allowed
            collections.

        Raises:
            asyncpg.InsufficientPrivilegeError: If the database role is missing a GRANT the
                query needs. A deployment fault rather than the caller's, so it stays a 500.
            asyncpg.QueryCanceledError: If the query runs past `DB_STATEMENT_TIMEOUT_MS`.
            TypeError: If a `collections` row has a column `Collection` does not mirror, as a
                migration that adds one ahead of the code would cause.
            ValueError: If a stored `vector_type` is not a known `VectorType`.
        """
        # nothing can be visible, so skip the round trip
        if not self.allowed_collection_names:
            return []

        async def _query(conn):
            stmt = await conn.prepare("SELECT * FROM collections WHERE id = ANY($1::bigint[])")
            rows = await stmt.fetch(collection_ids)
            return [Collection.from_record(r) for r in rows]

        return await self._with_rls(_query)

    async def count_available_embeddings_in_collection(self, collection: Collection) -> int:
        """
        Count the embeddings in a collection that are visible to this caller.

        The count is RLS-filtered, so it reflects what the caller can actually read rather
        than the true row count for the collection.

        Args:
            collection (Collection): Collection to count.

        Returns:
            int: Number of visible embeddings.

        Raises:
            asyncpg.InsufficientPrivilegeError: If the database role is missing a GRANT the
                query needs. A deployment fault rather than the caller's, so it stays a 500.
            asyncpg.QueryCanceledError: If the query runs past `DB_STATEMENT_TIMEOUT_MS`.
        """
        table, _ = get_embeddings_table_and_cast(VectorType(collection.vector_type))

        async def _query(conn):
            stmt = await conn.prepare(
                f"""
                SELECT COUNT(*) AS cnt
                FROM {table}
                WHERE collection_id = $1::bigint
                """
            )
            row = await stmt.fetchrow(collection.id)
            return row["cnt"] if row else 0

        return await self._with_rls(_query)
