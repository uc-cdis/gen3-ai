"""Data access methods that create, update and delete collections and embeddings."""

from dataclasses import dataclass
from typing import Any
from uuid import UUID

import asyncpg
import numpy as np
from asyncpg.exceptions import UniqueViolationError

from gen3_embeddings.database import hashing
from gen3_embeddings.database.dal.base import DataAccessLayerBase
from gen3_embeddings.database.errors import (
    CollectionAlreadyExistsError,
    CollectionCreateFailedError,
    CollectionNameNotAllowedError,
    DuplicateEmbeddingError,
    EmbeddingNotFoundError,
    EmbeddingsAlreadyExistError,
    EmbeddingWriteInconsistencyError,
    MetadataLengthMismatchError,
    RepeatedEmbeddingIdError,
)
from gen3_embeddings.database.helpers import affected_row_count, get_embeddings_table_and_cast
from gen3_embeddings.database.models import Collection, Embedding
from gen3_embeddings.models.schemas import VectorType


@dataclass(frozen=True)
class _BulkWriteBatch:
    """
    A deduplicated batch of embeddings, hashed and shaped for the bulk INSERT's parameters.

    Vectors travel as one flat float32 array that the INSERT slices per row, and the per-row
    columns travel as parallel arrays. That is what keeps the whole batch on asyncpg's binary
    encoding path: nothing here is a JSON document, so no float is ever formatted as text.
    """

    dimensions: int
    # len == row_count * dimensions, row-major
    flat_vectors: list[float]
    # canonical JSON text, one per unique row; also what the metadata hash was taken over
    metadata_json: list[str]
    embedding_hashes: list[UUID]
    metadata_hashes: list[UUID]
    # for each index in the caller's original list, which unique row it maps to
    original_to_unique: list[int]
    has_duplicates: bool

    @property
    def row_count(self) -> int:
        """Number of unique rows the INSERT will write."""
        return len(self.embedding_hashes)

    @property
    def row_keys(self) -> list[tuple[UUID, UUID]]:
        """
        The (embedding_hash, metadata_hash) pair identifying each unique row, in row order.

        Deduplication is by exactly this pair, so it is unique across the batch and usable to
        match RETURNING rows back to inputs without relying on the order Postgres emits them.
        """
        return list(zip(self.embedding_hashes, self.metadata_hashes))


def _prepare_bulk_write(
    collection: Collection,
    embeddings: list[list[float]],
    metadata_list: list[dict] | None,
) -> _BulkWriteBatch:
    """
    Hash a batch of embeddings and drop the duplicates within it.

    Deduplication is on (embedding_hash, metadata_hash); `authz` is a single value for the
    whole call, so it is constant within a batch and cannot distinguish rows. Because the
    hashes are taken at storage precision, two inputs that differ only in digits the column
    cannot store now collapse here, which is what the database's unique constraint would
    consider them anyway.

    Args:
        collection (Collection): Target collection; supplies dimensions and vector type.
        embeddings (list[list[float]]): Vectors to write.
        metadata_list (list[dict] | None): Metadata per vector, or None for all-empty.

    Returns:
        _BulkWriteBatch: The deduplicated batch, ready to bind.

    Raises:
        MetadataLengthMismatchError: If `metadata_list` is a different length than
            `embeddings`.
        EmbeddingDimensionMismatchError: If a vector's length is not the collection's
            dimensionality.
        EmbeddingNotRepresentableError: If a value cannot be stored in the collection's
            vector type.
        ValueError: If any metadata holds a NaN or Infinity value.
    """
    if metadata_list is None:
        metadata_list = [{} for _ in embeddings]
    elif len(metadata_list) != len(embeddings):
        raise MetadataLengthMismatchError("metadata_list length must match embeddings length")

    vector_type = VectorType(collection.vector_type)
    # one conversion for the whole batch; its rows are the bytes Postgres will store, which
    # is both what gets hashed and what gets bound
    array = hashing.to_storage_array(embeddings, vector_type, collection.dimensions)
    return _dedupe_rows(array, metadata_list, collection.dimensions)


def _dedupe_rows(array: np.ndarray, metadata_list: list[dict], dimensions: int) -> _BulkWriteBatch:
    """
    Hash already-converted rows and drop the duplicates among them.

    Args:
        array (np.ndarray): Storage-precision rows from `hashing.to_storage_array`.
        metadata_list (list[dict]): Metadata per row, the same length as `array`.
        dimensions (int): The collection's dimensionality.

    Returns:
        _BulkWriteBatch: The deduplicated batch, ready to bind.

    Raises:
        ValueError: If any metadata holds a NaN or Infinity value.
    """
    embedding_hashes = hashing.hash_rows(array)

    unique_row_indices: list[int] = []
    unique_metadata_json: list[str] = []
    unique_embedding_hashes: list[UUID] = []
    unique_metadata_hashes: list[UUID] = []
    # key -> unique row index
    seen: dict[tuple[UUID, UUID], int] = {}
    # for each original index i, which unique index j it maps to
    original_to_unique: list[int] = []
    has_duplicates = False

    for index, (embedding_hash, metadata) in enumerate(zip(embedding_hashes, metadata_list)):
        metadata_json = hashing.canonical_metadata_json(metadata)
        metadata_hash = hashing.hash_metadata_json(metadata_json)
        key = (embedding_hash, metadata_hash)

        if key in seen:
            original_to_unique.append(seen[key])
            has_duplicates = True
            continue

        unique_index = len(unique_row_indices)
        seen[key] = unique_index
        original_to_unique.append(unique_index)
        unique_row_indices.append(index)
        unique_metadata_json.append(metadata_json)
        unique_embedding_hashes.append(embedding_hash)
        unique_metadata_hashes.append(metadata_hash)

    return _BulkWriteBatch(
        dimensions=dimensions,
        flat_vectors=hashing.flatten_rows(array, unique_row_indices),
        metadata_json=unique_metadata_json,
        embedding_hashes=unique_embedding_hashes,
        metadata_hashes=unique_metadata_hashes,
        original_to_unique=original_to_unique,
        has_duplicates=has_duplicates,
    )


def _bulk_write_results(rows: list[asyncpg.Record], batch: _BulkWriteBatch) -> list[Embedding]:
    """
    Map the rows a bulk write returned back onto the caller's original input order.

    RETURNING order is not something Postgres promises, so rows are matched by the hash pair
    they came back with rather than by position. Every unique row has a distinct pair by
    construction, so the match is exact.

    Args:
        rows (list[asyncpg.Record]): Rows from the INSERT's RETURNING clause.
        batch (_BulkWriteBatch): The batch that was written.

    Returns:
        list[Embedding]: One Embedding per embedding the caller passed in, in that order.
            Inputs that deduplicated onto the same row share an object.

    Raises:
        EmbeddingWriteInconsistencyError: If the returned rows do not correspond exactly to
            the rows requested, which would mean a row was silently dropped or duplicated.
    """
    if len(rows) != batch.row_count:
        raise EmbeddingWriteInconsistencyError("Internal error: mismatch between unique upsert results and inputs.")

    rows_by_key = {(row["embedding_hash_v2"], row["metadata_hash_v2"]): row for row in rows}
    try:
        unique_results = [Embedding.from_record(rows_by_key[key]) for key in batch.row_keys]
    except KeyError as exc:
        raise EmbeddingWriteInconsistencyError(
            "Internal error: a written embedding could not be matched back to its input."
        ) from exc

    if not batch.has_duplicates:
        # nothing collapsed, so unique order is already the caller's order
        return unique_results

    return [unique_results[unique_index] for unique_index in batch.original_to_unique]


class WriteMixin(DataAccessLayerBase):
    """Inserts, upserts, updates and deletes. RLS policies decide which rows a write may touch."""

    async def create_collection(
        self,
        collection_name: str,
        description: str | None,
        dimensions: int,
        ai_model_name: str | None = None,
        vector_type: VectorType = VectorType.vector,
    ) -> Collection:
        """
        Create a collection, if the caller is allowed to use that name.

        The name check here is what turns "not authorized" into a specific error naming the
        collection, rather than the generic `RowLevelSecurityDeniedError` the table's RLS
        WITH CHECK would produce. The policy still applies underneath, so removing this check
        would change the error, not the outcome. Because both read the same set, a name that
        passes this check also passes the policy, which is why that error is not listed below.

        Args:
            collection_name (str): Name of the collection to create.
            description (str | None): Human-readable description; the column is nullable.
            dimensions (int): Vector dimensionality for embeddings in this collection.
            ai_model_name (str | None): Model the embeddings were produced with, if known.
            vector_type (VectorType): Storage type, `vector` (float32) or `halfvec` (float16).

        Returns:
            Collection: The newly created collection.

        Raises:
            CollectionNameNotAllowedError: If the caller may not use this collection name.
            CollectionAlreadyExistsError: If the name is already taken.
            CollectionCreateFailedError: If the insert returned no row.
            asyncpg.InsufficientPrivilegeError: If the database role is missing a GRANT the
                query needs. A deployment fault rather than the caller's, so it stays a 500.
            asyncpg.QueryCanceledError: If the query runs past `DB_STATEMENT_TIMEOUT_MS`.
            TypeError: If a `collections` row has a column `Collection` does not mirror, as a
                migration that adds one ahead of the code would cause.
            ValueError: If a stored `vector_type` is not a known `VectorType`.
        """
        if collection_name not in self.allowed_collection_names:
            raise CollectionNameNotAllowedError(f"Not authorized to create collection with name {collection_name}")

        async def _query(conn):
            try:
                stmt = await conn.prepare(
                    """
                    INSERT INTO collections (collection_name, description, ai_model_name, dimensions, vector_type)
                    VALUES ($1::text, $2::text, $3::text, $4::int, $5::text)
                    RETURNING *
                    """
                )
                row = await stmt.fetchrow(collection_name, description, ai_model_name, dimensions, vector_type.value)
            except UniqueViolationError:
                # collection_name already exists
                raise CollectionAlreadyExistsError(f"Collection '{collection_name}' already exists")
            if not row:
                raise CollectionCreateFailedError("Failed to create collection")
            return Collection.from_record(row)

        return await self._with_rls(_query)

    async def update_collection(self, collection_name: str, description: str | None) -> Collection | None:
        """
        Update a collection's mutable fields, if the caller is allowed to.

        Passing `description=None` updates nothing and simply returns the current row.

        `collection_name` is only ever a WHERE predicate here, never something this method
        assigns: the name is what the table's RLS policy keys on, so renaming a collection
        would move it to a different authz resource. There is no API path to do that.

        Args:
            collection_name (str): Name of the collection to update.
            description (str | None): New description, or None to leave it unchanged.

        Returns:
            Collection | None: The collection after the update, or None if it does not
            exist **or** RLS hid it from this caller.

        Raises:
            asyncpg.InsufficientPrivilegeError: If the database role is missing a GRANT the
                query needs. A deployment fault rather than the caller's, so it stays a 500.
            asyncpg.QueryCanceledError: If the query runs past `DB_STATEMENT_TIMEOUT_MS`.
            TypeError: If a `collections` row has a column `Collection` does not mirror, as a
                migration that adds one ahead of the code would cause.
            ValueError: If a stored `vector_type` is not a known `VectorType`.
        """
        set_parts = []
        params: list[Any] = [collection_name]
        param_idx = 2

        if description is not None:
            set_parts.append(f"description = ${param_idx}::text")
            params.append(description)
            param_idx += 1

        async def _query(conn):
            if not set_parts:
                # nothing to update
                stmt = await conn.prepare("SELECT * FROM collections WHERE collection_name = $1::text")
                row = await stmt.fetchrow(collection_name)
                return Collection.from_record(row) if row else None

            set_clause = ", ".join(set_parts) + ", updated_at = NOW()"

            stmt = await conn.prepare(
                f"""
                UPDATE collections
                SET {set_clause}
                WHERE collection_name = $1::text
                RETURNING *
                """
            )
            row = await stmt.fetchrow(*params)
            return Collection.from_record(row) if row else None

        return await self._with_rls(_query)

    async def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection and, by cascade, every embedding in it.

        The cascade is a referential-integrity action, which Postgres runs without applying
        RLS. So deleting a collection removes every embedding in it, including rows the
        caller could not have selected under its own `app.allowed_authz`. That is the point
        of the collection-level grant, and it is why `delete` on a collection is a
        meaningfully broader permission than `delete` on its embeddings.

        Args:
            collection_name (str): Name of the collection to delete.

        Returns:
            bool: True only if a row was actually deleted. False if RLS hid the collection
            from this caller, or if no collection by that name existed.

        Raises:
            asyncpg.InsufficientPrivilegeError: If the database role is missing a GRANT the
                query needs. A deployment fault rather than the caller's, so it stays a 500.
            asyncpg.QueryCanceledError: If the query runs past `DB_STATEMENT_TIMEOUT_MS`.
        """

        async def _query(conn):
            result = await conn.execute(
                "DELETE FROM collections WHERE collection_name = $1::text",
                collection_name,
            )
            return affected_row_count(result) > 0

        return await self._with_rls(_query)

    async def create_embeddings_bulk(
        self,
        collection: Collection,
        embeddings: list[list[float]],
        authz: str,
        metadata_list: list[dict] | None,
    ) -> list[Embedding]:
        """
        Bulk-insert embeddings into a collection; every input must be new.

        Args:
            collection (Collection): Target collection; supplies dimensions and vector type.
            embeddings (list[list[float]]): Vectors to insert.
            authz (str): Authz resource path assigned to every embedding in this batch.
            metadata_list (list[dict] | None): Metadata per vector, or None for all-empty.

        Returns:
            list[Embedding]: One Embedding per input vector, in input order. Inputs that
            deduplicated onto the same row within the batch share an object.

        Raises:
            MetadataLengthMismatchError: If `metadata_list` is a different length than
                `embeddings`.
            EmbeddingDimensionMismatchError: If any vector's length does not match the
                collection's dimensionality.
            EmbeddingNotRepresentableError: If a value cannot be stored in the collection's
                vector type.
            ValueError: If any metadata holds a NaN or Infinity value, which the jsonb column
                cannot store. The request schema refuses these first, so this is reached only
                by a caller that skipped it.
            EmbeddingsAlreadyExistError: If any embedding in the batch conflicts with an
                existing row. No embeddings are written.
            RowLevelSecurityDeniedError: If `authz` is not a resource the caller holds this
                action on. No embeddings are written.
            EmbeddingWriteInconsistencyError: If the rows returned by the database do not
                correspond exactly to the rows inserted.
            asyncpg.ForeignKeyViolationError: If the collection was deleted after the caller
                looked it up.
            asyncpg.InsufficientPrivilegeError: If the database role is missing a GRANT the
                query needs. A deployment fault rather than the caller's, so it stays a 500.
            asyncpg.QueryCanceledError: If the query runs past `DB_STATEMENT_TIMEOUT_MS`.
        """
        batch = _prepare_bulk_write(collection, embeddings, metadata_list)
        if not batch.row_count:
            return []

        table, cast = get_embeddings_table_and_cast(VectorType(collection.vector_type))

        async def _query(conn):
            # execute one concurrent safe query
            stmt = await conn.prepare(
                f"""
                INSERT INTO {table} (
                    collection_id, embedding, authz, metadata,
                    embedding_hash, metadata_hash, embedding_hash_v2, metadata_hash_v2
                )
                SELECT
                    $1::bigint,
                    -- one flat float4[] for the batch, sliced per row. asyncpg encodes it in
                    -- binary, so the vectors never become text on the way here.
                    ($2::float4[])[((raw.ord - 1) * $3::int + 1):(raw.ord * $3::int)]{cast},
                    $4::text,
                    raw.metadata::jsonb,
                    -- legacy md5 columns, written with the sha256 value so their NOT NULL and
                    -- unique constraint stay satisfied until the contract migration drops
                    -- them. See db/migrations/20260826120000_sha256_content_hashes.sql.
                    raw.embedding_hash,
                    raw.metadata_hash,
                    raw.embedding_hash,
                    raw.metadata_hash
                FROM unnest($5::text[], $6::uuid[], $7::uuid[])
                    WITH ORDINALITY AS raw(metadata, embedding_hash, metadata_hash, ord)
                -- the hashes come back so results can be matched to inputs by content rather
                -- than by an order Postgres does not guarantee
                RETURNING collection_id, embedding_id, embedding, authz, metadata, created_at, updated_at,
                          embedding_hash_v2, metadata_hash_v2;
                """
            )
            try:
                rows = await stmt.fetch(
                    collection.id,
                    batch.flat_vectors,
                    batch.dimensions,
                    authz,
                    batch.metadata_json,
                    batch.embedding_hashes,
                    batch.metadata_hashes,
                )
            except UniqueViolationError as exc:
                raise EmbeddingsAlreadyExistError(
                    "One or more embeddings already exist in this collection. "
                    "No embeddings were created. Use PUT to force update existing embeddings."
                ) from exc

            return _bulk_write_results(rows, batch)

        return await self._with_rls(_query)

    async def upsert_embeddings_bulk(
        self,
        collection: Collection,
        embeddings: list[list[float]],
        authz: str,
        metadata_list: list[dict] | None,
        embedding_ids: list[UUID | None] | None = None,
    ) -> list[Embedding]:
        """
        Write a batch of embeddings, updating the ones named by id and upserting the rest.

        An item with an id replaces that embedding's vector, metadata and authz; the id has to
        exist. An item without one is inserted, or matched to an existing row with the same
        content and authz, whose `updated_at` is refreshed. Both halves run in one transaction,
        so the batch is written in full or not at all.

        Every vector is checked before the transaction opens, against the whole batch, so the
        index in a validation error is the item's position in `embeddings`.

        Args:
            collection (Collection): Target collection; supplies dimensions and vector type.
            embeddings (list[list[float]]): Vectors to write.
            authz (str): Authz resource path assigned to every embedding in this batch.
            metadata_list (list[dict] | None): Metadata per vector, or None for all-empty.
            embedding_ids (list[UUID | None] | None): Per vector, the embedding it replaces, or
                None to upsert it by content. Omitted means every vector is upserted.

        Returns:
            list[Embedding]: One Embedding per input vector, in input order. Inputs that
            landed on the same row share it.

        Raises:
            ValueError: If `embedding_ids` is a different length than `embeddings`, or any
                metadata holds a NaN or Infinity value, which the jsonb column cannot store.
                The request schema refuses the latter first, so it is reached only by a caller
                that skipped it.
            MetadataLengthMismatchError: If `metadata_list` is a different length than
                `embeddings`.
            RepeatedEmbeddingIdError: If an id appears more than once. One UPDATE cannot apply
                two different writes to the same row, so it would keep one and drop the other.
            EmbeddingDimensionMismatchError: If any vector's length does not match the
                collection's dimensionality.
            EmbeddingNotRepresentableError: If a value cannot be stored in the collection's
                vector type.
            EmbeddingNotFoundError: If an id does not exist in the collection, or RLS hides it.
                Nothing is written.
            DuplicateEmbeddingError: If an update by id would collide with another row.
                Nothing is written.
            RowLevelSecurityDeniedError: If `authz` is not a resource the caller holds this
                action on. Nothing is written.
            EmbeddingWriteInconsistencyError: If the rows returned by the database do not
                correspond exactly to the rows written.
            asyncpg.ForeignKeyViolationError: If the collection was deleted after the caller
                looked it up.
            asyncpg.InsufficientPrivilegeError: If the database role is missing a GRANT the
                query needs. A deployment fault rather than the caller's, so it stays a 500.
            asyncpg.QueryCanceledError: If the query runs past `DB_STATEMENT_TIMEOUT_MS`.
        """
        if embedding_ids is None:
            embedding_ids = [None] * len(embeddings)
        elif len(embedding_ids) != len(embeddings):
            raise ValueError("embedding_ids and embeddings must be the same length")
        if metadata_list is None:
            metadata_list = [{} for _ in embeddings]
        elif len(metadata_list) != len(embeddings):
            raise MetadataLengthMismatchError("metadata_list length must match embeddings length")
        if not embeddings:
            return []

        seen: set[UUID] = set()
        for embedding_id in embedding_ids:
            if embedding_id is None:
                continue
            if embedding_id in seen:
                raise RepeatedEmbeddingIdError(f"embedding_id {embedding_id} appears more than once in this request")
            seen.add(embedding_id)

        # One conversion for the whole batch, before the transaction opens: a bad vector is the
        # caller's error, not a reason to have taken a connection, and checking both halves
        # together is what keeps the index in the error the caller's own.
        vector_type = VectorType(collection.vector_type)
        array = hashing.to_storage_array(embeddings, vector_type, collection.dimensions)
        by_id = [i for i, embedding_id in enumerate(embedding_ids) if embedding_id is not None]
        by_content = [i for i, embedding_id in enumerate(embedding_ids) if embedding_id is None]

        update_ids = [embedding_ids[i] for i in by_id]
        update_vectors = hashing.flatten_rows(array, by_id)
        update_embedding_hashes = hashing.hash_rows(array[by_id])
        update_metadata_json = [hashing.canonical_metadata_json(metadata_list[i]) for i in by_id]
        update_metadata_hashes = [hashing.hash_metadata_json(text) for text in update_metadata_json]

        batch = _dedupe_rows(array[by_content], [metadata_list[i] for i in by_content], collection.dimensions)

        table, cast = get_embeddings_table_and_cast(vector_type)

        async def _query(conn):
            written: dict[int, Embedding] = {}

            # Updates by id go first, so an id-less item with the same content as an updated row
            # finds that row's new content and resolves onto it rather than colliding.
            if by_id:
                stmt = await conn.prepare(
                    f"""
                    UPDATE {table} AS e
                    SET
                        -- the same flat float4[] slicing as the bulk INSERT, so vectors stay binary
                        embedding = ($2::float4[])[((u.ord - 1) * $3::int + 1):(u.ord * $3::int)]{cast},
                        authz = $4::text,
                        metadata = u.metadata::jsonb,
                        -- legacy md5 columns, see the note in create_embeddings_bulk
                        embedding_hash = u.embedding_hash,
                        metadata_hash = u.metadata_hash,
                        embedding_hash_v2 = u.embedding_hash,
                        metadata_hash_v2 = u.metadata_hash,
                        updated_at = NOW()
                    FROM unnest($5::text[], $6::uuid[], $7::uuid[], $8::uuid[])
                        WITH ORDINALITY AS u(metadata, embedding_id, embedding_hash, metadata_hash, ord)
                    WHERE e.collection_id = $1::bigint AND e.embedding_id = u.embedding_id
                    RETURNING e.collection_id, e.embedding_id, e.embedding, e.authz, e.metadata,
                              e.created_at, e.updated_at, u.ord
                    """
                )
                try:
                    rows = await stmt.fetch(
                        collection.id,
                        update_vectors,
                        collection.dimensions,
                        authz,
                        update_metadata_json,
                        update_ids,
                        update_embedding_hashes,
                        update_metadata_hashes,
                    )
                except UniqueViolationError as exc:
                    raise DuplicateEmbeddingError(
                        "Update would create a duplicate embedding "
                        "with same vector, metadata, and authz in this collection."
                    ) from exc

                for row in rows:
                    written[by_id[row["ord"] - 1]] = Embedding.from_record(row)

                missing = [embedding_id for embedding_id, i in zip(update_ids, by_id) if i not in written]
                if missing:
                    # Raised inside the transaction, so everything written so far rolls back.
                    shown = ", ".join(str(embedding_id) for embedding_id in missing[:5])
                    more = f" and {len(missing) - 5} more" if len(missing) > 5 else ""
                    raise EmbeddingNotFoundError(f"No embedding with id {shown}{more} in this collection")

            if batch.row_count:
                stmt = await conn.prepare(
                    f"""
                    INSERT INTO {table} (
                        collection_id, embedding, authz, metadata,
                        embedding_hash, metadata_hash, embedding_hash_v2, metadata_hash_v2
                    )
                    SELECT
                        $1::bigint,
                        ($2::float4[])[((raw.ord - 1) * $3::int + 1):(raw.ord * $3::int)]{cast},
                        $4::text,
                        raw.metadata::jsonb,
                        -- legacy md5 columns, see the note in create_embeddings_bulk
                        raw.embedding_hash,
                        raw.metadata_hash,
                        raw.embedding_hash,
                        raw.metadata_hash
                    FROM unnest($5::text[], $6::uuid[], $7::uuid[])
                        WITH ORDINALITY AS raw(metadata, embedding_hash, metadata_hash, ord)
                    -- Conflicts resolve on the v2 index. A new row writes the same value to the
                    -- legacy columns, so anything that would collide there collides here too and
                    -- is handled; a collision with a legacy md5 value would need sha256 and md5
                    -- to agree, which is not a case worth carrying code for.
                    ON CONFLICT (collection_id, embedding_hash_v2, metadata_hash_v2, authz)
                    DO UPDATE SET
                        updated_at = NOW()
                    RETURNING collection_id, embedding_id, embedding, authz, metadata, created_at, updated_at,
                              embedding_hash_v2, metadata_hash_v2;
                    """
                )
                # If RLS denies insert or update, this will raise an error
                rows = await stmt.fetch(
                    collection.id,
                    batch.flat_vectors,
                    batch.dimensions,
                    authz,
                    batch.metadata_json,
                    batch.embedding_hashes,
                    batch.metadata_hashes,
                )
                for position, embedding in zip(by_content, _bulk_write_results(rows, batch)):
                    written[position] = embedding

            return [written[i] for i in range(len(embeddings))]

        return await self._with_rls(_query)

    async def update_embedding(
        self,
        collection: Collection,
        embedding_id: UUID,
        embedding: list[float] | None,
        metadata: dict | None,
        new_authz: str | None = None,
    ) -> Embedding | None:
        """
        Update an embedding row in the appropriate embeddings_* table.

        - If `embedding` is provided, update the vector and recompute embedding_hash.
        - If `metadata` is provided, update metadata and recompute metadata_hash.
        - If `new_authz` is provided, update authz.

        The combination (collection_id, embedding_hash_v2, metadata_hash_v2, authz)
        must remain unique (per the DB constraint).

        Args:
            collection (Collection): Collection the embedding belongs to.
            embedding_id (UUID): Identifier of the embedding to update.
            embedding (list[float] | None): New vector, or None to leave it unchanged.
            metadata (dict | None): New metadata, or None to leave it unchanged.
            new_authz (str | None): New authz resource path, or None to leave it unchanged.

        Returns:
            Embedding | None: The updated row, or None if no such embedding is visible to the caller.

        Raises:
            DuplicateEmbeddingError: If the update would collide with another row.
            EmbeddingDimensionMismatchError: If `embedding` is not the collection's
                dimensionality.
            EmbeddingNotRepresentableError: If `embedding` holds a value the collection's
                vector type cannot store.
            ValueError: If `metadata` holds a NaN or Infinity value, which the jsonb column
                cannot store. The request schema refuses these first, so this is reached only
                by a caller that skipped it.
            RowLevelSecurityDeniedError: If `new_authz` is not a resource the caller holds
                this action on.
            asyncpg.InsufficientPrivilegeError: If the database role is missing a GRANT the
                query needs. A deployment fault rather than the caller's, so it stays a 500.
            asyncpg.QueryCanceledError: If the query runs past `DB_STATEMENT_TIMEOUT_MS`.
        """
        vector_type = VectorType(collection.vector_type)
        table, vector_cast = get_embeddings_table_and_cast(vector_type)

        # hash before opening the transaction; a bad vector is the caller's error, not a
        # reason to have taken a connection out of the pool
        embedding_hash = (
            hashing.hash_vector(embedding, vector_type, collection.dimensions) if embedding is not None else None
        )
        # the same canonical text is both stored and hashed, so this row's hash matches what
        # a bulk write of identical metadata would produce
        metadata_json = hashing.canonical_metadata_json(metadata) if metadata is not None else None
        metadata_hash = hashing.hash_metadata_json(metadata_json) if metadata_json is not None else None

        async def _query(conn):
            set_parts = []
            params = [collection.id, embedding_id]
            param_idx = 3

            # embedding: update vector and embedding_hash. The vector binds natively (pgvector
            # registers a binary codec on the pool), so it is never serialized to text.
            if embedding is not None:
                set_parts.append(f"embedding = ${param_idx}{vector_cast}")
                params.append(embedding)
                param_idx += 1

                # legacy md5 column gets the sha256 value too; see
                # db/migrations/20260826120000_sha256_content_hashes.sql
                set_parts.append(f"embedding_hash = ${param_idx}::uuid")
                set_parts.append(f"embedding_hash_v2 = ${param_idx}::uuid")
                params.append(embedding_hash)
                param_idx += 1

            # metadata: update metadata and metadata_hash
            if metadata is not None:
                set_parts.append(f"metadata = ${param_idx}::jsonb")
                params.append(metadata_json)
                param_idx += 1

                set_parts.append(f"metadata_hash = ${param_idx}::uuid")
                set_parts.append(f"metadata_hash_v2 = ${param_idx}::uuid")
                params.append(metadata_hash)
                param_idx += 1

            # authz: update authz
            if new_authz is not None:
                set_parts.append(f"authz = ${param_idx}::text")
                params.append(new_authz)
                param_idx += 1

            if not set_parts:
                # nothing to update; just read and return the existing row
                stmt = await conn.prepare(
                    f"""
                    SELECT *
                    FROM {table}
                    WHERE collection_id = $1::bigint AND embedding_id = $2::uuid
                    """
                )
                row = await stmt.fetchrow(collection.id, embedding_id)
                return Embedding.from_record(row) if row else None

            set_clause = ", ".join(set_parts) + ", updated_at = NOW()"

            stmt = await conn.prepare(
                f"""
                UPDATE {table}
                SET {set_clause}
                WHERE collection_id = $1::bigint AND embedding_id = $2::uuid
                RETURNING *
                """
            )
            try:
                row = await stmt.fetchrow(*params)
            except UniqueViolationError as exc:
                # updating caused a collision with another row that has the same
                # (collection_id, embedding_hash_v2, metadata_hash_v2, authz)
                raise DuplicateEmbeddingError(
                    "Update would create a duplicate embedding "
                    "with same vector, metadata, and authz in this collection."
                ) from exc

            return Embedding.from_record(row) if row else None

        return await self._with_rls(_query)

    async def delete_embedding(
        self,
        collection: Collection,
        embedding_id: UUID,
    ) -> bool:
        """
        Delete a single embedding from a collection.

        Args:
            collection (Collection): Collection the embedding belongs to.
            embedding_id (UUID): Identifier of the embedding to delete.

        Returns:
            bool: True only if a row was actually deleted. False if no such embedding
            existed in the collection, or if RLS hid it from this caller.

        Raises:
            asyncpg.InsufficientPrivilegeError: If the database role is missing a GRANT the
                query needs. A deployment fault rather than the caller's, so it stays a 500.
            asyncpg.QueryCanceledError: If the query runs past `DB_STATEMENT_TIMEOUT_MS`.
        """
        table, _ = get_embeddings_table_and_cast(VectorType(collection.vector_type))

        async def _query(conn):
            result = await conn.execute(
                f"DELETE FROM {table} WHERE collection_id = $1::bigint AND embedding_id = $2::uuid",
                collection.id,
                embedding_id,
            )
            return affected_row_count(result) > 0

        return await self._with_rls(_query)
