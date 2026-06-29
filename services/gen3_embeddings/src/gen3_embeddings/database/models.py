import json
from dataclasses import dataclass
from datetime import datetime
from typing import Self
from uuid import UUID

import asyncpg
from pgvector import HalfVector, Vector

from gen3_embeddings.models.schemas import VectorType


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
