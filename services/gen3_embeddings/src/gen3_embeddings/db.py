from __future__ import annotations

import ast
import json
from dataclasses import dataclass, fields
from datetime import datetime
from uuid import UUID

import asyncpg
from fastapi import HTTPException, Request

from gen3_embeddings import config
from gen3_embeddings.auth import _get_crud_action_from_request, get_allowed_authz_from_mapping, get_user_authz_mapping
from gen3_embeddings.config import logging

_pool: asyncpg.Pool | None = None


async def get_pool():
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(str(config.DB_CONNECTION_STRING), min_size=1, max_size=10)
    return _pool


async def get_data_access_layer():
    pool = await get_pool()
    yield DataAccessLayer(pool)


@dataclass
class VectorIndex:
    id: int
    vector_index_name: str
    description: str | None
    ai_model_name: str | None
    dimensions: int
    created_at: datetime | None
    updated_at: datetime | None

    @classmethod
    def from_record(cls, row: asyncpg.Record) -> VectorIndex:
        data = dict(row)
        return cls(**data)


@dataclass
class Embedding:
    vector_index_id: int
    embedding_id: UUID
    embedding: list[float]
    authz_version: int
    authz: list[str]
    metadata: dict | None
    created_at: datetime | None
    updated_at: datetime | None

    @classmethod
    def from_record(cls, row: asyncpg.Record) -> Embedding:
        data = dict(row)

        # Keep only keys that match Embedding fields
        valid_field_names = {f.name for f in fields(cls)}
        data = {k: v for k, v in data.items() if k in valid_field_names}

        if isinstance(data.get("embedding"), str):
            # vec_str is like "[0.1, 0.2, 0.3]", convert it to vector
            data["embedding"] = [float(x) for x in ast.literal_eval(data["embedding"])]
        return cls(**data)


class DataAccessLayer:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def _with_rls(self, request: Request, fn, *args, **kwargs):
        """
        Run a DB operation with RLS (row level security) set according to user's allowed authz.

        This will:
        - get user's authz mapping
        - compute allowed authz tags for this CRUD operation
        - SET LOCAL app.allowed_authz
        then run `fn(conn, *args, **kwargs)` with the same transaction scope.
        """
        user_authz_mapping = await get_user_authz_mapping(request=request)
        method = _get_crud_action_from_request(request)
        allowed_authz = get_allowed_authz_from_mapping(
            authz_mapping=user_authz_mapping,
            method=method,
        )
        logging.debug(f"allowed_authz for {method}: {allowed_authz}")

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Make RLS safe even if user has no allowed resources
                allowed_array = "{" + ",".join(allowed_authz) + "}"
                await conn.execute("SELECT set_config('app.allowed_authz', $1, true)", allowed_array)
                return await fn(conn, *args, **kwargs)

    async def create_vector_index(
        self, index_name: str, description: str, dimensions: int, ai_model_name: str | None = None
    ) -> VectorIndex:
        async with self.pool.acquire() as conn:
            stmt = await conn.prepare("""
                INSERT INTO vector_indices (vector_index_name, description, ai_model_name, dimensions)
                VALUES ($1, $2, $3, $4)
                RETURNING *
            """)
            row = await stmt.fetchrow(index_name, description, ai_model_name, dimensions)
            if not row:
                raise HTTPException(status_code=400, detail="Failed to create vector index")
            return VectorIndex.from_record(row)

    async def get_vector_index_by_name(self, index_name: str) -> VectorIndex | None:
        async with self.pool.acquire() as conn:
            stmt = await conn.prepare("SELECT * FROM vector_indices WHERE vector_index_name = $1")
            row = await stmt.fetchrow(index_name)
            return VectorIndex.from_record(row) if row else None

    async def get_vector_index_by_id(self, index_id: int) -> VectorIndex | None:
        async with self.pool.acquire() as conn:
            stmt = await conn.prepare("SELECT * FROM vector_indices WHERE id = $1")
            row = await stmt.fetchrow(index_id)
            return VectorIndex.from_record(row) if row else None

    async def update_vector_index(self, index_name: str, update_fields: dict) -> VectorIndex | None:
        keys, values = zip(*update_fields.items())
        set_clause = ", ".join([f"{k} = ${i + 2}" for i, k in enumerate(keys)])
        async with self.pool.acquire() as conn:
            stmt = await conn.prepare(
                f"UPDATE vector_indices SET {set_clause}, updated_at = NOW() WHERE vector_index_name = $1 RETURNING *"
            )
            row = await stmt.fetchrow(index_name, *values)
            return VectorIndex.from_record(row) if row else None

    async def delete_vector_index(self, index_name: str) -> bool:
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM vector_indices WHERE vector_index_name = $1",
                index_name,
            )
            return result.startswith("DELETE")

    async def list_vector_indices(self) -> list[VectorIndex]:
        async with self.pool.acquire() as conn:
            stmt = await conn.prepare("SELECT * FROM vector_indices ORDER BY created_at")
            rows = await stmt.fetch()
            return [VectorIndex.from_record(r) for r in rows]

    async def create_embedding(
        self,
        vector_index_id: int,
        embedding: list[float],
        authz_version: int,
        authz: list[str],
        metadata: dict | None = None,
    ) -> Embedding:
        async with self.pool.acquire() as conn:
            stmt = await conn.prepare("""
                INSERT INTO embeddings
                (vector_index_id, embedding, authz_version, authz, metadata)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING *
            """)
            row = await stmt.fetchrow(vector_index_id, embedding, authz_version, authz, metadata or {})
            if not row:
                raise HTTPException(status_code=400, detail="Failed to create embedding")
            return Embedding.from_record(row)

    async def create_embeddings_bulk(
        self,
        request: Request,
        vector_index_id: int,
        embeddings: list[list[float]],
        authz_version: int,
        authz: list[str],
        metadata_list: list[dict] | None = None,
    ) -> list[Embedding]:
        """
        TODO: embeddings need to have same dim?
        TODO: why emb_vec and meta have to be string? and the vector return from
        TODO: asyncpg.exceptions.InsufficientPrivilegeError: new row violates row-level security policy for table "embeddings"
        the database is string instead of list of float? Current temp fix is convert
        them to accepted format.

        Bulk create multiple embeddings in the given index.

        Args:
            vector_index_id: ID of the vector index to insert into.
            embeddings: List of embedding vectors.
            authz_version: Authorization schema version.
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

        # Use a single connection and transaction for all inserts
        async def _query(conn):
            results: list[Embedding] = []
            for emb_vec, meta in zip(embeddings, metadata_list):
                row = await conn.fetchrow(
                    """
                    INSERT INTO embeddings
                    (vector_index_id, embedding, authz_version, authz, metadata)
                    VALUES ($1, $2, $3, $4, $5)
                    RETURNING *
                    """,
                    vector_index_id,
                    json.dumps(emb_vec),
                    authz_version,
                    authz,
                    json.dumps(meta or {}),
                )
                if not row:
                    raise HTTPException(
                        status_code=400,
                        detail="Failed to create embedding in bulk insert",
                    )
                results.append(Embedding.from_record(row))
            return results

        return await self._with_rls(request, _query)

    async def get_embedding_by_id(self, request: Request, embedding_id: UUID) -> Embedding | None:
        async def _query(conn):
            stmt = await conn.prepare("SELECT * FROM embeddings WHERE embedding_id = $1")
            row = await stmt.fetchrow(embedding_id)
            return Embedding.from_record(row) if row else None

        return await self._with_rls(request, _query)

    async def get_embedding_by_index_and_id(
        self, request: Request, vector_index_id: int, embedding_id: UUID
    ) -> Embedding | None:
        async def _query(conn):
            stmt = await conn.prepare("SELECT * FROM embeddings WHERE vector_index_id = $1 AND embedding_id = $2")
            row = await stmt.fetchrow(vector_index_id, embedding_id)
            return Embedding.from_record(row) if row else None

        return await self._with_rls(request, _query)

    async def update_embedding(
        self, request: Request, vector_index_id: int, embedding_id: UUID, embedding: list[float]
    ) -> Embedding | None:
        # TODO: embedding has to be string currently, look into why.
        async def _query(conn):
            stmt = await conn.prepare("""
                UPDATE embeddings SET embedding = $3, updated_at = NOW()
                WHERE vector_index_id = $1 AND embedding_id = $2
                RETURNING *
            """)
            row = await stmt.fetchrow(vector_index_id, embedding_id, json.dumps(embedding))
            return Embedding.from_record(row) if row else None

        return await self._with_rls(request, _query)

    async def delete_embedding(self, request: Request, vector_index_id: int, embedding_id: UUID) -> bool:
        async def _query(conn):
            result = await conn.execute(
                "DELETE FROM embeddings WHERE vector_index_id = $1 AND embedding_id = $2", vector_index_id, embedding_id
            )
            return result.startswith("DELETE")

        return await self._with_rls(request, _query)

    async def list_embeddings_in_index(self, request: Request, vector_index_id: int) -> list[Embedding]:
        async def _query(conn):
            stmt = await conn.prepare("SELECT * FROM embeddings WHERE vector_index_id = $1 ORDER BY created_at")
            rows = await stmt.fetch(vector_index_id)
            return [Embedding.from_record(r) for r in rows]

        return await self._with_rls(request, _query)

    async def get_embeddings_bulk(self, request: Request, embedding_ids: list[UUID]) -> list[Embedding]:
        async def _query(conn):
            stmt = await conn.prepare("SELECT * FROM embeddings WHERE embedding_id = ANY($1::uuid[])")
            rows = await stmt.fetch(embedding_ids)
            return [Embedding.from_record(r) for r in rows]

        return await self._with_rls(request, _query)

    async def get_vector_index_by_id_bulk(self, index_ids: list[int]) -> list[VectorIndex]:
        async with self.pool.acquire() as conn:
            stmt = await conn.prepare("SELECT * FROM vector_indices WHERE id = ANY($1::bigint[])")
            rows = await stmt.fetch(index_ids)
            return [VectorIndex.from_record(r) for r in rows]

    # -------- Search --------

    async def search_embeddings_in_index(
        self,
        request: Request,
        vector_index_id: int,
        query_vector: list[float],
        top_k: int = 10,
        score_range: float | None = None,
        filters: dict[str, str] | None = None,
    ) -> list[asyncpg.Record]:
        """
        TODO: more search  algorithm or methods
        TODO: embedding/query_vector has to be string currently, look into why.

        Minimal search implementation using pgvector and cosine distance.
        Returns raw rows with extra 'similarity_score' column.
        """
        filters = filters or {}
        where_clauses = ["vector_index_id = $1"]
        params = [vector_index_id, json.dumps(query_vector), top_k]

        param_index = 4
        for k, v in filters.items():
            where_clauses.append(f"metadata->>$${k}$$ = ${param_index}")
            params.append(v)
            param_index += 1

        where_sql = " AND ".join(where_clauses)
        # cosine distance: 1 - (embedding <=> query)
        # assuming pgvector <=> is cosine distance; adjust if using L2.
        sql = f"""
            SELECT *,
                   1 - (embedding <=> $2::vector) AS similarity_score
            FROM embeddings
            WHERE {where_sql}
            ORDER BY embedding <=> $2::vector
            LIMIT $3
        """

        async def _query(conn):
            stmt = await conn.prepare(sql)
            rows = await stmt.fetch(*params)
            if score_range is not None:
                rows = [r for r in rows if r["similarity_score"] >= score_range]
            return rows

        return await self._with_rls(request, _query)

    async def search_embeddings_across_indices(
        self,
        request: Request,
        vector_index_ids: list[int],
        query_vector: list[float],
        top_k: int = 10,
        score_range: float | None = None,
        filters: dict[str, str] | None = None,
    ) -> list[asyncpg.Record]:
        # TODO: embedding/query_vector has to be string currently, look into why.
        # TODO: currently this cannot handle diff dim since this is searching all indices
        # whcih can hanve diff dim
        if not vector_index_ids:
            return []

        filters = filters or {}
        params = [vector_index_ids, json.dumps(query_vector), top_k]
        where_clauses = ["vector_index_id = ANY($1::bigint[])"]
        param_index = 4

        for k, v in filters.items():
            where_clauses.append(f"metadata->>$${k}$$ = ${param_index}")
            params.append(v)
            param_index += 1

        where_sql = " AND ".join(where_clauses)
        sql = f"""
            SELECT *,
                   1 - (embedding <=> $2::vector) AS similarity_score
            FROM embeddings
            WHERE {where_sql}
            ORDER BY embedding <=> $2::vector
            LIMIT $3
        """

        async def _query(conn):
            stmt = await conn.prepare(sql)
            rows = await stmt.fetch(*params)
            if score_range is not None:
                rows = [r for r in rows if r["similarity_score"] >= score_range]
            return rows

        return await self._with_rls(request, _query)
