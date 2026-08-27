"""
This file houses the database logic.

OVERVIEW
--------

We're using asyncpg alongside FastAPI's dependency injection.

This file contains the logic for database manipulation in a "data access layer"
(DataAccessLayer / DAL) class, such that other areas of the code have simple
`.create_*()`, `.list_*()`, `.search_*()` calls which won't require knowledge
of how to manage connections or interact with the db directly. Connections are
managed via an asyncpg connection pool and FastAPI's dependency injection
provides a DAL instance per-request.

Each DAL instance is bound to the caller's authz resources for RLS purposes, but it does
not resolve or interpret them: `gen3_embeddings.dependencies` resolves authz at the HTTP
boundary and hands the results in. Nothing in this file talks to anything but the database,
and nothing here raises HTTP errors - see `database/errors.py`.

DETAILS
-------

What do we do in this file?

- We create an asyncpg connection pool as a module-level global
    - The pool is initialized once (on demand) using the DB URL from config

- We define lightweight dataclasses for Collections and Embeddings
    - These mirror rows from the database and provide `.from_record()` helpers
      to convert from asyncpg.Record objects

- We define a DataAccessLayer class which isolates all database manipulations
    - All CRUD and search operations go through this interface instead of
      leaking raw SQL into the higher-level web app endpoint code
    - DAL methods use prepared statements where appropriate as a security and
      efficiency measure
    - Each DAL instance carries a per-request `allowed_authz` list, derived
      from the current user's Arborist authz mapping

- We are deliberately free of HTTP and authz concerns
    - DAL methods raise the domain errors in `database/errors.py`; the mapping to status
      codes lives in `gen3_embeddings.error_handlers`
    - The FastAPI dependencies that build a DAL live in `gen3_embeddings.dependencies`,
      because resolving authz requires a network call to the Gen3 policy engine and this
      layer should only ever talk to the database

- We implement Row Level Security (RLS) integration for embeddings tables
    - Before each logical operation on embeddings, `_with_rls()` sets a
      per-transaction PostgreSQL parameter `app.allowed_authz` using:
          SELECT set_config('app.allowed_authz', $1, true);
      where `$1` is a text representation of the user's allowed authz resources
    - Because `set_config(..., true)` uses a local/transaction-scoped setting,
      each request runs with its own authz context even when using a pooled
      connection
    - The embeddings tables (e.g., `embeddings_vector`, `embeddings_halfvec`)
      define RLS policies that consult `current_setting('app.allowed_authz', true)`

- We support multiple vector types as isolated domains
    - Collections store a `vector_type` (e.g., 'vector', 'halfvec')
    - Each vector type has its own embeddings table (e.g., `embeddings_vector`,
      `embeddings_halfvec`)
    - DAL methods route all embedding CRUD and search operations to the
      appropriate table based on the collection's `vector_type`
    - Search methods use pgvector operators and functions and expose
      a uniform interface with configurable distance metrics, min/max thresholds,
      and filters on metadata

- We filter collections by an explicitly supplied set of names
    - Collections themselves do not have RLS or authz tags stored in the table
    - So collection methods take an `allowed_collection_names` argument, resolved by the
      route layer. This layer filters by that set but never decides what belongs in it;
      an empty set means "no collections", which is fail-closed
"""

from dataclasses import dataclass
from typing import Any
from uuid import UUID

import asyncpg
from asyncpg.exceptions import UniqueViolationError
from pgvector.asyncpg import register_vector

from gen3_embeddings import config
from gen3_embeddings.config import logging
from gen3_embeddings.database import hashing
from gen3_embeddings.database.errors import (
    CollectionAlreadyExistsError,
    CollectionCreateFailedError,
    CollectionNameNotAllowedError,
    DuplicateEmbeddingError,
    EmbeddingsAlreadyExistError,
    EmbeddingWriteInconsistencyError,
    InvalidCollectionNameError,
    MetadataLengthMismatchError,
)
from gen3_embeddings.database.helpers import affected_row_count, build_search_sql, get_embeddings_table_and_cast
from gen3_embeddings.database.models import Collection, Embedding
from gen3_embeddings.models.helpers import normalize_collection_name
from gen3_embeddings.models.schemas import DistanceMetric, VectorType

_pool: asyncpg.Pool | None = None


async def get_pool():
    """
    Gets the pool of connections.

    We have a special initialization to support pgvector columns efficiently.

    See https://github.com/pgvector/pgvector-python

    The `register_vector` adds a custom codec for the `vector` and `halfvec` column types.

    This ensures that when we read from/write to pgvector, it happens
    at the binary level rather than as a string for maximum efficiency.

    Without this, asyncpg defaults to treating it like a string - which is incredibly
    inefficient b/c that's not how it's stored.
    """
    global _pool
    if _pool is None:
        logging.info(
            "Initializing connection pool... pool min=%d, pool max=%d", config.PGPOOL_MIN_SIZE, config.PGPOOL_MAX_SIZE
        )
        _pool = await asyncpg.create_pool(
            str(config.DB_CONNECTION_STRING),
            min_size=config.PGPOOL_MIN_SIZE,
            max_size=config.PGPOOL_MAX_SIZE,
            init=register_vector,
        )
    return _pool


async def close_pool() -> None:
    """
    Close the global connection pool and reset it.

    Safe to call when the pool was never created.
    """
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


@dataclass(frozen=True)
class _BulkWriteBatch:
    """
    A deduplicated batch of embeddings, hashed and shaped for the bulk INSERT's parameters.

    Vectors travel as one flat float4[] that the INSERT slices per row, and the per-row
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
    """
    if metadata_list is None:
        metadata_list = [{} for _ in embeddings]
    elif len(metadata_list) != len(embeddings):
        raise MetadataLengthMismatchError("metadata_list length must match embeddings length")

    vector_type = VectorType(collection.vector_type)
    # one conversion for the whole batch; its rows are the bytes Postgres will store, which
    # is both what gets hashed and what gets bound
    array = hashing.to_storage_array(embeddings, vector_type, collection.dimensions)
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
        dimensions=collection.dimensions,
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


class DataAccessLayer:
    """
    Database interface for collections and embeddings, scoped to one caller's authz.

    Each instance carries the authz resources the caller may act on for the current
    request, which embedding operations hand to Postgres row-level security via `_with_rls`.
    The `collections` table has no RLS policy, so collection methods instead filter by an
    `allowed_collection_names` set that the route layer resolves and passes in.
    """

    def __init__(self, pool: asyncpg.Pool, allowed_authz: list[str] | None = None):
        """
        Bind the DAL to a connection pool and the caller's allowed authz resources.

        Args:
            pool (asyncpg.Pool): Shared connection pool.
            allowed_authz (list[str] | None): Authz resource paths the caller may act on.
                Omitted or empty means no resources are allowed, which is a valid
                fail-closed state rather than "allow everything".
        """
        self.pool = pool
        # Empty list means "no resources allowed", which is valid for RLS
        self.allowed_authz = allowed_authz or []

    async def _with_rls(self, fn, *args, **kwargs):
        """
        Run a DB operation with RLS (row level security) configured
        via the provided `allowed_authz` values.

        The caller (route layer) is responsible for:
        - determining the allowed_authz list from the user's authz mapping.
        """
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # the true value means is_local is set to true, the new value will only apply during the current transaction.
                await conn.execute("SELECT set_config('app.allowed_authz', $1::text[]::text, true)", self.allowed_authz)
                return await fn(conn, *args, **kwargs)

    async def create_collection(
        self,
        collection_name: str,
        description: str | None,
        dimensions: int,
        allowed_collection_names: set[str],
        ai_model_name: str | None = None,
        vector_type: VectorType = VectorType.vector,
    ) -> Collection:
        """
        Create a collection, if the caller is allowed to use that name.

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
        """
        if collection_name not in allowed_collection_names:
            raise CollectionNameNotAllowedError(f"Not authorized to create collection with name {collection_name}")

        async with self.pool.acquire() as conn:
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

    async def get_collection_by_name(
        self, collection_name: str, allowed_collection_names: set[str]
    ) -> Collection | None:
        """
        Look up a collection by name, if the caller is allowed to see it.

        Args:
            collection_name (str): Name of the collection; normalized before lookup.

        Returns:
            Collection | None: The collection, or None if it does not exist **or** the
            caller is not authorized for it. The two cases are deliberately
            indistinguishable so callers cannot probe for collection names; callers
            typically surface this as a 404.

        Raises:
            InvalidCollectionNameError: If `collection_name` is not a valid collection name.
        """
        try:
            collection_name = normalize_collection_name(collection_name)
        except ValueError as exc:
            raise InvalidCollectionNameError(str(exc)) from exc

        if collection_name not in allowed_collection_names:
            return None

        async with self.pool.acquire() as conn:
            stmt = await conn.prepare("SELECT * FROM collections WHERE collection_name = $1::text")
            row = await stmt.fetchrow(collection_name)
            return Collection.from_record(row) if row else None

    async def get_collection_by_id(self, collection_id: int, allowed_collection_names: set[str]) -> Collection | None:
        """
        Look up a collection by primary key, if the caller is allowed to see it.

        Args:
            collection_id (int): Primary key of the collection.

        Returns:
            Collection | None: The collection, or None if it does not exist **or** the
            caller is not authorized for it.
        """
        if not allowed_collection_names:
            return None
        async with self.pool.acquire() as conn:
            stmt = await conn.prepare("SELECT * FROM collections WHERE id = $1::bigint")
            row = await stmt.fetchrow(collection_id)
            if row and row.collection_name in allowed_collection_names:
                return Collection.from_record(row)
            else:
                return None

    async def update_collection(
        self, collection_name: str, description: str | None, allowed_collection_names: set[str]
    ) -> Collection | None:
        """
        Update a collection's mutable fields, if the caller is allowed to.

        Passing `description=None` updates nothing and simply returns the current row.

        Args:
            collection_name (str): Name of the collection to update.
            description (str | None): New description, or None to leave it unchanged.

        Returns:
            Collection | None: The collection after the update, or None if it does not
            exist **or** the caller is not authorized for it.
        """
        if collection_name not in allowed_collection_names:
            return None

        set_parts = []
        params = [collection_name]
        param_idx = 2

        if description is not None:
            set_parts.append(f"description = ${param_idx}::text")
            params.append(description)
            param_idx += 1

        async with self.pool.acquire() as conn:
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

    async def delete_collection(self, collection_name: str, allowed_collection_names: set[str]) -> bool:
        """
        Delete a collection and, by cascade, every embedding in it.

        Args:
            collection_name (str): Name of the collection to delete.

        Returns:
            bool: True only if a row was actually deleted. False if the caller is not
            authorized for this collection, or if no collection by that name existed.
        """
        if collection_name not in allowed_collection_names:
            return False

        async with self.pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM collections WHERE collection_name = $1::text",
                collection_name,
            )
            return affected_row_count(result) > 0

    async def list_collections(
        self,
        allowed_collection_names: set[str],
        offset: int = 0,
        limit: int = 100,
    ) -> list[Collection]:
        """
        List the collections the caller is authorized for.

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
        """
        # If no allowed names, return empty result
        if not allowed_collection_names:
            return []

        async with self.pool.acquire() as conn:
            stmt = await conn.prepare(
                """
                SELECT *
                FROM collections
                WHERE collection_name = ANY($1::text[])
                ORDER BY collection_name
                LIMIT $3::int
                OFFSET $2::int
                """
            )
            rows = await stmt.fetch(list(allowed_collection_names), offset, limit)
            return [Collection.from_record(r) for r in rows]

    async def create_embeddings_bulk(
        self,
        collection: Collection,
        embeddings: list[list[float]],
        authz: str,
        metadata_list: list[dict] | None,
    ) -> list[Embedding]:
        """
        Bulk create multiple embeddings in the given collection.

        Args:
            collection: collection to insert into.
            embeddings: List of embedding vectors.
            authz: Authorization tags.
            metadata_list: Optional list of metadata dicts (one per embedding).

        Returns:
            List of created Embedding instances.
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
        """
        table, _ = get_embeddings_table_and_cast(VectorType(collection.vector_type))

        async def _query(conn):
            stmt = await conn.prepare(
                f"SELECT * FROM {table} WHERE collection_id = $1::bigint AND embedding_id = $2::uuid"
            )
            row = await stmt.fetchrow(collection.id, embedding_id)
            return Embedding.from_record(row) if row else None

        return await self._with_rls(_query)

    async def upsert_embeddings_bulk(
        self,
        collection: Collection,
        embeddings: list[list[float]],
        authz: str,
        metadata_list: list[dict] | None,
    ) -> list[Embedding]:
        """
        Bulk upsert multiple embeddings in the given collection.
        """
        batch = _prepare_bulk_write(collection, embeddings, metadata_list)
        if not batch.row_count:
            return []

        table, cast = get_embeddings_table_and_cast(VectorType(collection.vector_type))

        async def _query(conn):
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

            return _bulk_write_results(rows, batch)

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

        Raises:
            DuplicateEmbeddingError: If the update would collide with another row.
            EmbeddingDimensionMismatchError: If `embedding` is not the collection's
                dimensionality.
            EmbeddingNotRepresentableError: If `embedding` holds a value the collection's
                vector type cannot store.
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
        Fetch embeddings by IDs from the appropriate table(s).

        If vector_type is given, only that table is queried.
        If None, both tables are queried and results combined.
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

    async def get_collection_by_id_bulk(
        self, collection_ids: list[int], allowed_collection_names: set[str]
    ) -> list[Collection]:
        """
        Fetch several collections by primary key, keeping only those the caller may see.

        Args:
            collection_ids (list[int]): Primary keys to look up.

        Returns:
            list[Collection]: Authorized collections, in no guaranteed order. May be
            shorter than `collection_ids`, and empty if the caller has no allowed
            collections.
        """
        # If user has no allowed collection names, return empty
        if not allowed_collection_names:
            return []

        async with self.pool.acquire() as conn:
            stmt = await conn.prepare("SELECT * FROM collections WHERE id = ANY($1::bigint[])")
            rows = await stmt.fetch(collection_ids)
            filtered_rows = [row for row in rows if row["collection_name"] in allowed_collection_names]
            return [Collection.from_record(r) for r in filtered_rows]

    # -------- Search --------

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
        Search embeddings within a single collection using the collection's vector_type.
        """
        table, cast = get_embeddings_table_and_cast(VectorType(collection.vector_type))

        # $1: collection_id, $2: vector, $3: top_k
        params: list[Any] = [collection.id, query_vector, top_k]

        sql, extra_params = build_search_sql(
            table=table,
            distance_metric=distance_metric,
            single_collection=True,
            collection_ids_param="$1::bigint",
            vector_param=f"$2{cast}",
            top_k_param="$3::int",
            filters=filters,
            min_value=min_value,
            max_value=max_value,
        )
        params.extend(extra_params)

        async def _query(conn):
            stmt = await conn.prepare(sql)
            rows = await stmt.fetch(*params)
            return rows

        return await self._with_rls(_query)

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
        """
        if not collections:
            return []

        # Filter collections by vector_type and dimensions
        filtered_collections: list[Collection] = []
        query_dims = len(query_vector)

        for col in collections:
            if col.vector_type == vector_type.value and col.dimensions == query_dims:
                filtered_collections.append(col)

        if not filtered_collections:
            return []

        table, cast = get_embeddings_table_and_cast(vector_type)
        collection_ids = [col.id for col in filtered_collections]

        # $1: collection_ids, $2: vector, $3: top_k
        params: list[Any] = [collection_ids, query_vector, top_k]

        sql, extra_params = build_search_sql(
            table=table,
            distance_metric=distance_metric,
            single_collection=False,
            collection_ids_param="$1::bigint[]",
            vector_param=f"$2{cast}",
            top_k_param="$3::int",
            filters=filters,
            min_value=min_value,
            max_value=max_value,
        )
        params.extend(extra_params)

        async def _query(conn):
            stmt = await conn.prepare(sql)
            rows = await stmt.fetch(*params)
            return rows

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
