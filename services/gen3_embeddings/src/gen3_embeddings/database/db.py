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

import ast
import json
from dataclasses import dataclass, fields
from datetime import datetime
from typing import Any, Self
from uuid import UUID

import asyncpg
from asyncpg.exceptions import UniqueViolationError
from fastapi import HTTPException, Request

from gen3_embeddings import config
from gen3_embeddings.auth import get_allowed_authz_for_request
from gen3_embeddings.database.helpers import build_search_sql, get_embeddings_table_and_cast
from gen3_embeddings.models.schemas import DistanceMetric, VectorType

_pool: asyncpg.Pool | None = None


async def get_pool():
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(str(config.DB_CONNECTION_STRING), min_size=10, max_size=10)
    return _pool


async def get_data_access_layer(request: Request):
    pool = await get_pool()
    allowed_authz = await get_allowed_authz_for_request(request)
    dal = DataAccessLayer(pool, allowed_authz=allowed_authz)
    yield dal


@dataclass
class Collection:
    id: int
    collection_name: str
    description: str | None
    ai_model_name: str | None
    dimensions: int
    vector_type: str
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
    embedding: list[float]
    authz: list[str]
    metadata: dict | None
    created_at: datetime | None
    updated_at: datetime | None

    @classmethod
    def from_record(cls, row: asyncpg.Record) -> Self:
        """
        Build an Embedding dataclass from an asyncpg.Record.

        This normalizes:
        - embedding: string → list[float]
        - metadata:  string → dict (JSON)
        """
        data = dict(row)

        # Keep only keys that match Embedding fields
        valid_field_names = {f.name for f in fields(cls)}
        data = {k: v for k, v in data.items() if k in valid_field_names}

        if isinstance(data.get("embedding"), str):
            # vec_str is like "[0.1, 0.2, 0.3]", convert it to vector
            data["embedding"] = [float(x) for x in ast.literal_eval(data["embedding"])]

        # metadata: convert string → dict if needed
        if isinstance(data.get("metadata"), str):
            try:
                data["metadata"] = json.loads(data["metadata"])
            except Exception:
                # fallback: keep it as None or an empty dict
                data["metadata"] = None

        return cls(**data)


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

    def _get_embeddings_table_and_cast_for_collection(self, collection: Collection) -> tuple[str, str]:
        return get_embeddings_table_and_cast(VectorType(collection.vector_type))

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
        vector_type: str = "vector",
    ) -> Collection:
        allowed_names = self.get_allowed_collection_names_from_allowed_authz()

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
                row = await stmt.fetchrow(collection_name, description, ai_model_name, dimensions, vector_type)
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
        allowed_names = self.get_allowed_collection_names_from_allowed_authz()

        if collection_name not in allowed_names:
            return None

        async with self.pool.acquire() as conn:
            stmt = await conn.prepare("SELECT * FROM collections WHERE collection_name = $1::text")
            row = await stmt.fetchrow(collection_name)
            return Collection.from_record(row) if row else None

    async def get_collection_by_id(self, collection_id: int) -> Collection | None:
        allowed_names = self.get_allowed_collection_names_from_allowed_authz()
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
        allowed_names = self.get_allowed_collection_names_from_allowed_authz()

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

    # async def delete_collection(self, collection_name: str) -> bool:
    #     async with self.pool.acquire() as conn:
    #         result = await conn.execute(
    #             "DELETE FROM collections WHERE collection_name = $1",
    #             collection_name,
    #         )
    #         return result.startswith("DELETE")

    async def delete_collection(self, collection_name: str) -> bool:
        allowed_names = self.get_allowed_collection_names_from_allowed_authz()

        if collection_name not in allowed_names:
            return False

        async with self.pool.acquire() as conn:
            stmt = await conn.prepare("DELETE FROM collections WHERE collection_name = $1::text")
            result = await stmt.execute(collection_name)
            return result.startswith("DELETE")

    async def list_collections(
        self,
        offset: int = 0,
        limit: int = 100,
    ) -> list[Collection]:
        allowed_names = self.get_allowed_collection_names_from_allowed_authz()

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
        TODO: embeddings need to have same dim?
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

        table, cast = self._get_embeddings_table_and_cast_for_collection(collection)

        async def _query(conn):
            results: list[Embedding] = []
            for emb_vec, meta in zip(embeddings, metadata_list):
                row = await conn.fetchrow(
                    f"""
                    INSERT INTO {table}
                    (collection_id, embedding, authz, metadata)
                    VALUES ($1::bigint, $2{cast}, $3::text[], $4::jsonb)
                    RETURNING *
                    """,
                    collection.id,
                    emb_vec,
                    authz,
                    meta or {},
                )
                if not row:
                    raise HTTPException(
                        status_code=400,
                        detail="Failed to create embedding in bulk insert",
                    )
                results.append(Embedding.from_record(row))
            return results

        return await self._with_rls(_query)

    async def get_embedding_by_collection_and_id(
        self,
        collection: Collection,
        embedding_id: UUID,
    ) -> Embedding | None:
        table, _ = self._get_embeddings_table_and_cast_for_collection(collection)

        async def _query(conn):
            stmt = await conn.prepare(
                f"SELECT * FROM {table} WHERE collection_id = $1::bigint AND embedding_id = $2::uuid"
            )
            row = await stmt.fetchrow(collection.id, embedding_id)
            return Embedding.from_record(row) if row else None

        return await self._with_rls(_query)

    async def update_embedding(
        self,
        collection: Collection,
        embedding_id: UUID,
        embedding: list[float] | None,
        metadata: dict | None,
    ) -> Embedding | None:
        # TODO: embedding has to be string currently, look into why.
        table, vector_cast = self._get_embeddings_table_and_cast_for_collection(collection)

        async def _query(conn):
            set_parts = []
            params = [collection.id, embedding_id]
            param_idx = 3

            if embedding is not None:
                set_parts.append(f"embedding = ${param_idx}{vector_cast}")
                # params.append(json.dumps(embedding))
                params.append(embedding)
                param_idx += 1

            if metadata is not None:
                set_parts.append(f"metadata = ${param_idx}::jsonb")
                # params.append(json.dumps(metadata))
                params.append(metadata)
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

    # async def delete_embedding(
    #     self,
    #     collection: Collection,
    #     embedding_id: UUID,
    #     allowed_authz: list[str],
    # ) -> bool:
    #     table, _ = self._get_embeddings_table_and_cast_for_collection(collection)

    #     async def _query(conn):
    #         result = await conn.execute(
    #             f"DELETE FROM {table} WHERE collection_id = $1 AND embedding_id = $2", collection.id, embedding_id
    #         )
    #         return result.startswith("DELETE")

    #     return await self._with_rls(allowed_authz, _query)

    async def delete_embedding(
        self,
        collection: Collection,
        embedding_id: UUID,
    ) -> bool:
        table, _ = self._get_embeddings_table_and_cast_for_collection(collection)

        async def _query(conn):
            stmt = await conn.prepare(
                f"DELETE FROM {table} WHERE collection_id = $1::bigint AND embedding_id = $2::uuid"
            )
            result = await stmt.execute(collection.id, embedding_id)
            return result.startswith("DELETE")

        return await self._with_rls(_query)

    async def list_embeddings_in_collection(
        self,
        collection: Collection,
        offset: int,
        limit: int,
    ) -> list[Embedding]:
        table, _ = self._get_embeddings_table_and_cast_for_collection(collection)

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
                stmt = await conn.prepare(f"SELECT * FROM {table} WHERE embedding_id = ANY($1::uuid[])")
                rows = await stmt.fetch(embedding_ids)
                results.extend(rows_to_embeddings(rows))
            else:
                # query both vector and halfvec tables
                for vt in (VectorType.vector, VectorType.halfvec):
                    table, _ = get_embeddings_table_and_cast(vt)
                    stmt = await conn.prepare(f"SELECT * FROM {table} WHERE embedding_id = ANY($1::uuid[])")
                    rows = await stmt.fetch(embedding_ids)
                    results.extend(rows_to_embeddings(rows))

            return results

        return await self._with_rls(_query)

    async def get_collection_by_id_bulk(self, collection_ids: list[int]) -> list[Collection]:
        async with self.pool.acquire() as conn:
            stmt = await conn.prepare("SELECT * FROM collections WHERE id = ANY($1::bigint[])")
            rows = await stmt.fetch(collection_ids)
            return [Collection.from_record(r) for r in rows]

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
            vector_param=f"$2{cast}",  # e.g. $2::vector or $2::halfvec
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
    ) -> list[asyncpg.Record]:
        """
        Search embeddings across multiple collections of the SAME vector_type.
        """
        if not collections:
            return []

        # Ensure they all share the same vector_type
        vec_type = collections[0].vector_type
        for col in collections:
            if col.vector_type != vec_type:
                raise HTTPException(
                    status_code=400,
                    detail="All collections must share the same vector_type for a single search",
                )

        table, cast = get_embeddings_table_and_cast(VectorType(vec_type))
        collection_ids = [col.id for col in collections]

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
