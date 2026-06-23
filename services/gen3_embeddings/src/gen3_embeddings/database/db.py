"""
This file houses the database logic.

OVERVIEW
--------

We're using asyncpg alongside FastAPI's dependency injection.

This file contains the logic for database manipulation in a "data access layer"
(DataAccessLayer) class, such that other areas of the code have simple
`.create_*()`, `.list_*()`, `.search_*()` calls which won't require knowledge
of how to manage connections or interact with the db directly. Connections are
managed via an asyncpg connection pool and FastAPI's dependency injection
provides a DAL instance per-request.

Each DAL instance is also bound to the current user's authorization context:
the caller's allowed authz resources are computed once (via the request) and
stored on the DAL instance, so downstream DAL methods do not need to be passed
`allowed_authz` explicitly.

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

- We provide a `get_data_access_layer()` function which yields an instance
  of the DAL bound to:
    - the global asyncpg connection pool, and
    - the current request's `allowed_authz` values
  This function is used as a FastAPI dependency, so each request handler
  can receive a DAL instance without managing connections or authz context
  manually.

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

- We apply authz-aware filtering for collections at the application layer
    - Collections themselves do not have RLS or authz tags stored in the table
    - Instead, the DAL derives allowed collection names from the per-request
      `allowed_authz` paths (e.g., "/vectorstore/collections/{collection_name}")
      and filters collection-level queries accordingly where needed
"""

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Self
from uuid import UUID

import asyncpg
from asyncpg.exceptions import UniqueViolationError
from fastapi import HTTPException, Request
from pgvector import HalfVector, Vector
from pgvector.asyncpg import register_vector

from gen3_embeddings import config
from gen3_embeddings.auth import get_allowed_authz_for_request, get_allowed_authz_for_request_with_method
from gen3_embeddings.database.helpers import build_search_sql, get_embeddings_table_and_cast
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
        _pool = await asyncpg.create_pool(
            str(config.DB_CONNECTION_STRING), min_size=10, max_size=10, init=register_vector
        )
    return _pool


async def get_data_access_layer(request: Request):
    pool = await get_pool()
    allowed_authz = await get_allowed_authz_for_request(request)
    dal = DataAccessLayer(pool, allowed_authz=allowed_authz)
    yield dal


async def get_data_access_layer_for_read_operations(request: Request):
    pool = await get_pool()
    allowed_authz = await get_allowed_authz_for_request_with_method(request, method="read")
    dal = DataAccessLayer(pool, allowed_authz=allowed_authz)
    yield dal


@dataclass
class Collection:
    id: int
    collection_name: str
    description: str | None
    ai_model_name: str | None
    dimensions: int
    vector_type: VectorType
    created_at: datetime | None
    updated_at: datetime | None

    @classmethod
    def from_record(cls, row: asyncpg.Record) -> Self:
        data = dict(row)
        return cls(**data)


@dataclass
class Embedding:
    collection_id: int
    embedding_id: UUID
    embedding: Vector | HalfVector
    authz: list[str]
    metadata: dict | None
    created_at: datetime | None
    updated_at: datetime | None

    @classmethod
    def from_record(cls, row: asyncpg.Record) -> Self:
        """
        Build an Embedding dataclass from an asyncpg.Record.

        This normalizes:
        - metadata:  string -> dict (JSON)
        """
        return cls(
            collection_id=row["collection_id"],
            embedding_id=row["embedding_id"],
            embedding=row["embedding"],
            authz=row["authz"],
            metadata=json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


class DataAccessLayer:
    def __init__(self, pool: asyncpg.Pool, allowed_authz: list[str] | None = None):
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
                # Make RLS safe even if user has no allowed resources
                allowed_array = "{" + ",".join(self.allowed_authz) + "}"
                # the true value means is_local is set to true, the new value will only apply during the current transaction.
                await conn.execute("SELECT set_config('app.allowed_authz', $1::text, true)", allowed_array)
                return await fn(conn, *args, **kwargs)

    def _get_allowed_collection_names_from_allowed_authz(self) -> set[str]:
        """
        Compute the collection names the user can access from self.allowed_authz,
        based on the convention:

          /vectorstore/collections
          /vectorstore/collections/{collection_name}
        """
        base = "/vectorstore/collections"
        allowed: set[str] = set()

        for item in self.allowed_authz:
            if not isinstance(item, str):
                continue
            if item == base:
                # base resource: may mean "can access all collections", depending on policy
                # for now, we'll pass
                continue
            if item.startswith(base + "/"):
                # e.g. "/vectorstore/collections/my_collection"
                parts = item.split("/")
                if parts:
                    if len(parts) != 4:
                        # # Expect exactly: ["", "vectorstore", "collections", "{collection_name}"]
                        # This covers "/vectorstore/collections/a/b" (len=5), etc.
                        continue
                    name = parts[-1]
                    if name:
                        allowed.add(name)
        return allowed

    async def create_collection(
        self,
        collection_name: str,
        description: str,
        dimensions: int,
        ai_model_name: str | None = None,
        vector_type: VectorType = VectorType.vector,
    ) -> Collection:
        allowed_names = self._get_allowed_collection_names_from_allowed_authz()

        if collection_name not in allowed_names:
            raise HTTPException(
                status_code=400, detail=f"Not authorized to create collection with name {collection_name}"
            )

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
                raise HTTPException(
                    status_code=400,
                    detail=f"Collection '{collection_name}' already exists",
                )
            if not row:
                raise HTTPException(status_code=400, detail="Failed to create collection")
            return Collection.from_record(row)

    async def get_collection_by_name(self, collection_name: str) -> Collection | None:
        allowed_names = self._get_allowed_collection_names_from_allowed_authz()

        if collection_name not in allowed_names:
            return None

        async with self.pool.acquire() as conn:
            stmt = await conn.prepare("SELECT * FROM collections WHERE collection_name = $1::text")
            row = await stmt.fetchrow(collection_name)
            return Collection.from_record(row) if row else None

    async def get_collection_by_id(self, collection_id: int) -> Collection | None:
        allowed_names = self._get_allowed_collection_names_from_allowed_authz()
        if not allowed_names:
            return None
        async with self.pool.acquire() as conn:
            stmt = await conn.prepare("SELECT * FROM collections WHERE id = $1::bigint")
            row = await stmt.fetchrow(collection_id)
            if row and row.collection_name in allowed_names:
                return Collection.from_record(row)
            else:
                return None

    async def update_collection(self, collection_name: str, description: str | None) -> Collection | None:
        allowed_names = self._get_allowed_collection_names_from_allowed_authz()

        if collection_name not in allowed_names:
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

    async def delete_collection(self, collection_name: str) -> bool:
        allowed_names = self._get_allowed_collection_names_from_allowed_authz()

        if collection_name not in allowed_names:
            return False

        async with self.pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM collections WHERE collection_name = $1::text",
                collection_name,
            )
            return result.startswith("DELETE")

    # async def delete_collection(self, collection_name: str) -> bool:
    #     allowed_names = self._get_allowed_collection_names_from_allowed_authz()

    #     if collection_name not in allowed_names:
    #         return False

    #     async with self.pool.acquire() as conn:
    #         stmt = await conn.prepare("DELETE FROM collections WHERE collection_name = $1::text")
    #         result = await stmt.fetch(collection_name)
    #         return result.startswith("DELETE")

    async def list_collections(
        self,
        offset: int = 0,
        limit: int = 100,
    ) -> list[Collection]:
        allowed_names = self._get_allowed_collection_names_from_allowed_authz()

        # If no allowed names, return empty result
        if not allowed_names:
            return []

        async with self.pool.acquire() as conn:
            stmt = await conn.prepare(
                """
                SELECT *
                FROM collections
                WHERE collection_name = ANY($1::text[])
                ORDER BY created_at
                LIMIT $3::int
                OFFSET $2::int
                """
            )
            rows = await stmt.fetch(list(allowed_names), offset, limit)
            return [Collection.from_record(r) for r in rows]

    async def create_embeddings_bulk(
        self,
        collection: Collection,
        embeddings: list[list[float]],
        authz: list[str],
        metadata_list: list[dict] | None,
    ) -> list[Embedding]:
        """
        TODO: why emb_vec and meta have to be string? and the vector return from
        TODO: asyncpg.exceptions.InsufficientPrivilegeError: new row violates row-level security policy for table "embeddings"
        the database is string instead of list of float? Current temp fix is convert
        them to accepted format.

        Bulk create multiple embeddings in the given collection.

        Args:
            collection: collection to insert into.
            embeddings: List of embedding vectors.
            authz: Authorization tags.
            metadata_list: Optional list of metadata dicts (one per embedding).

        Returns:
            List of created Embedding instances.
        """
        if metadata_list is None:
            metadata_list = [{} for _ in embeddings]
        elif len(metadata_list) != len(embeddings):
            raise HTTPException(
                status_code=400,
                detail="metadata_list length must match embeddings length",
            )

        table, cast = get_embeddings_table_and_cast(VectorType(collection.vector_type))

        async def _query(conn):
            """
            Convert data in bulk to JSON, then use postgres support for JSON -> records
            to bulk insert in a single transaction.
            """
            payload_data = [
                {
                    "collection_id": collection.id,
                    "embedding": json.dumps(emb_vec),
                    "authz": authz,
                    "metadata": metadata or {},
                }
                for emb_vec, metadata in zip(embeddings, metadata_list)
            ]

            # convert the entire batch into a single JSON string
            json_payload = json.dumps(payload_data)

            # execute one concurrent safe query
            stmt = await conn.prepare(
                f"""
                INSERT INTO {table} (collection_id, embedding, authz, metadata)
                SELECT
                    raw.collection_id,
                    raw.embedding{cast},
                    raw.authz,
                    raw.metadata
                FROM jsonb_to_recordset($1::jsonb) AS raw(
                    collection_id bigint,
                    embedding text,
                    authz text[],
                    metadata jsonb
                )
                -- only return what we need back, not the embedding itself b/c it's big
                -- and we already have it in Python
                RETURNING embedding_id, created_at, updated_at;
                """
            )
            try:
                rows = await stmt.fetch(json_payload)
            except UniqueViolationError as exc:
                raise HTTPException(
                    status_code=409,
                    detail="One or more embeddings already exist in this collection. "
                    "No embeddings were created. Use PUT to force update existing embeddings.",
                ) from exc

            results = [
                Embedding(
                    embedding_id=row["embedding_id"],
                    collection_id=collection.id,
                    embedding=emb_vec,
                    authz=authz,
                    metadata=meta,
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
                for row, emb_vec, meta in zip(rows, embeddings, metadata_list)
            ]

            return results

        return await self._with_rls(_query)

    async def get_embedding_by_collection_and_id(
        self,
        collection: Collection,
        embedding_id: UUID,
    ) -> Embedding | None:
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
        authz: list[str],
        metadata_list: list[dict] | None,
    ) -> list[Embedding]:
        """
        Bulk upsert multiple embeddings in the given collection.

        - If an embedding (same collection_id + vector) does not exist, it is inserted.
        - If it exists, it is updated (metadata/authz).
        - Entire operation is transactional (all or nothing).
        """
        if metadata_list is None:
            metadata_list = [{} for _ in embeddings]
        elif len(metadata_list) != len(embeddings):
            raise HTTPException(
                status_code=400,
                detail="metadata_list length must match embeddings length",
            )

        table, cast = get_embeddings_table_and_cast(VectorType(collection.vector_type))

        async def _query(conn):
            payload_data = [
                {
                    "collection_id": collection.id,
                    "embedding": json.dumps(emb_vec),
                    "authz": authz,
                    "metadata": metadata or {},
                }
                for emb_vec, metadata in zip(embeddings, metadata_list)
            ]

            json_payload = json.dumps(payload_data)

            # Important: ON CONFLICT on (collection_id, embedding)
            # Update authz and metadata, plus updated_at.
            # RLS WITH CHECK enforces that the user has permission to modify rows.
            stmt = await conn.prepare(
                f"""
                INSERT INTO {table} (collection_id, embedding, authz, metadata)
                SELECT
                    raw.collection_id,
                    raw.embedding{cast},
                    raw.authz,
                    raw.metadata
                FROM jsonb_to_recordset($1::jsonb) AS raw(
                    collection_id bigint,
                    embedding text,
                    authz text[],
                    metadata jsonb
                )
                ON CONFLICT (collection_id, embedding)
                DO UPDATE SET
                    authz = EXCLUDED.authz,
                    metadata = EXCLUDED.metadata,
                    updated_at = NOW()
                RETURNING collection_id, embedding_id, embedding, authz, metadata, created_at, updated_at;
                """
            )
            # If RLS denies insert or update, this will raise an error
            rows = await stmt.fetch(json_payload)

            results = [Embedding.from_record(row) for row in rows]
            return results

        return await self._with_rls(_query)

    async def update_embedding(
        self,
        collection: Collection,
        embedding_id: UUID,
        embedding: list[float] | None,
        metadata: dict | None,
        new_authz: list[str] | None = None,
    ) -> Embedding | None:
        # TODO: embedding has to be string currently, look into why.
        table, vector_cast = get_embeddings_table_and_cast(VectorType(collection.vector_type))

        async def _query(conn):
            set_parts = []
            params = [collection.id, embedding_id]
            param_idx = 3

            if embedding is not None:
                set_parts.append(f"embedding = ${param_idx}{vector_cast}")
                params.append(json.dumps(embedding))
                # params.append(embedding)
                param_idx += 1

            if metadata is not None:
                set_parts.append(f"metadata = ${param_idx}::jsonb")
                params.append(json.dumps(metadata))
                # params.append(metadata)
                param_idx += 1

            if new_authz is not None:
                set_parts.append(f"authz = ${param_idx}::text[]")
                params.append(new_authz)
                param_idx += 1

            if not set_parts:
                # nothing to update
                stmt = await conn.prepare(
                    f"SELECT * FROM {table} WHERE collection_id = $1::bigint AND embedding_id = $2::uuid"
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
            row = await stmt.fetchrow(*params)
            return Embedding.from_record(row) if row else None

        return await self._with_rls(_query)

    async def delete_embedding(
        self,
        collection: Collection,
        embedding_id: UUID,
    ) -> bool:
        table, _ = get_embeddings_table_and_cast(VectorType(collection.vector_type))

        async def _query(conn):
            result = await conn.execute(
                f"DELETE FROM {table} WHERE collection_id = $1::bigint AND embedding_id = $2::uuid",
                collection.id,
                embedding_id,
            )
            return result.startswith("DELETE")

        return await self._with_rls(_query)

    # async def delete_embedding(
    #     self,
    #     collection: Collection,
    #     embedding_id: UUID,
    # ) -> bool:
    #     table, _ = get_embeddings_table_and_cast(VectorType(collection.vector_type))

    #     async def _query(conn):
    #         stmt = await conn.prepare(
    #             f"DELETE FROM {table} WHERE collection_id = $1::bigint AND embedding_id = $2::uuid"
    #         )
    #         result = await stmt.execute(collection.id, embedding_id)
    #         return result.startswith("DELETE")

    #     return await self._with_rls(_query)

    async def list_embeddings_in_collection(
        self,
        collection: Collection,
        offset: int,
        limit: int,
    ) -> list[Embedding]:
        table, _ = get_embeddings_table_and_cast(VectorType(collection.vector_type))

        async def _query(conn):
            stmt = await conn.prepare(
                f"""
                SELECT * FROM {table}
                WHERE collection_id = $1::bigint
                ORDER BY created_at
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

    async def get_collection_by_id_bulk(self, collection_ids: list[int]) -> list[Collection]:
        allowed_names = self._get_allowed_collection_names_from_allowed_authz()

        # If user has no allowed collection names, return empty
        if not allowed_names:
            return []

        async with self.pool.acquire() as conn:
            stmt = await conn.prepare("SELECT * FROM collections WHERE id = ANY($1::bigint[])")
            rows = await stmt.fetch(collection_ids)
            filtered_rows = [row for row in rows if row["collection_name"] in allowed_names]
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
            # No collections meet the requested vector_type and dimension
            raise HTTPException(
                status_code=400,
                detail=(
                    "No collections available with the requested vector_type "
                    f"'{vector_type.value}' and dimensions {query_dims}"
                ),
            )

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
