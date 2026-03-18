from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

import asyncpg
from fastapi import HTTPException

from gen3_embeddings import config

_pool: asyncpg.Pool | None = None


async def get_pool():
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(str(config.DB_CONNECTION_STRING), min_size=1, max_size=10)
    return _pool


async def get_embedding_dal():
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
        return cls(**dict(row))


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
        return cls(**data)


class DataAccessLayer:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

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

    async def get_vector_index(self, index_name: str) -> VectorIndex | None:
        async with self.pool.acquire() as conn:
            stmt = await conn.prepare("SELECT * FROM vector_indices WHERE vector_index_name = $1")
            row = await stmt.fetchrow(index_name)
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
            stmt = await conn.prepare("DELETE FROM vector_indices WHERE vector_index_name = $1")
            result = await stmt.execute(index_name)
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

    async def get_embedding_by_id(self, embedding_id: UUID) -> Embedding | None:
        async with self.pool.acquire() as conn:
            stmt = await conn.prepare("SELECT * FROM embeddings WHERE embedding_id = $1")
            row = await stmt.fetchrow(embedding_id)
            return Embedding.from_record(row) if row else None

    async def get_embedding_by_index_and_id(self, vector_index_id: int, embedding_id: UUID) -> Embedding | None:
        async with self.pool.acquire() as conn:
            stmt = await conn.prepare("SELECT * FROM embeddings WHERE vector_index_id = $1 AND embedding_id = $2")
            row = await stmt.fetchrow(vector_index_id, embedding_id)
            return Embedding.from_record(row) if row else None

    async def update_embedding(
        self, vector_index_id: int, embedding_id: UUID, embedding: list[float]
    ) -> Embedding | None:
        async with self.pool.acquire() as conn:
            stmt = await conn.prepare("""
                UPDATE embeddings SET embedding = $3, updated_at = NOW()
                WHERE vector_index_id = $1 AND embedding_id = $2
                RETURNING *
            """)
            row = await stmt.fetchrow(vector_index_id, embedding_id, embedding)
            return Embedding.from_record(row) if row else None

    async def delete_embedding(self, vector_index_id: int, embedding_id: UUID) -> bool:
        async with self.pool.acquire() as conn:
            stmt = await conn.prepare("DELETE FROM embeddings WHERE vector_index_id = $1 AND embedding_id = $2")
            result = await stmt.execute(vector_index_id, embedding_id)
            return result.startswith("DELETE")

    async def list_embeddings_in_index(self, vector_index_id: int) -> list[Embedding]:
        async with self.pool.acquire() as conn:
            stmt = await conn.prepare("SELECT * FROM embeddings WHERE vector_index_id = $1 ORDER BY created_at")
            rows = await stmt.fetch(vector_index_id)
            return [Embedding.from_record(r) for r in rows]

    async def get_embeddings_bulk(self, embedding_ids: list[UUID]) -> list[Embedding]:
        async with self.pool.acquire() as conn:
            stmt = await conn.prepare("SELECT * FROM embeddings WHERE embedding_id = ANY($1::uuid[])")
            rows = await stmt.fetch(embedding_ids)
            return [Embedding.from_record(r) for r in rows]
