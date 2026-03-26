from __future__ import annotations

from uuid import UUID

from gen3_embeddings.db import Embedding, VectorIndex
from gen3_embeddings.models.schemas import (
    EmbeddingInfo,
    EmbeddingInfoNoIndex,
    SingleEmbeddingResult,
    SingleEmbeddingResultNoIndex,
    VectorIndexModel,
)


def build_embedding_self_url(index_name: str | None, embedding_id: UUID) -> str:
    """
    Build the 'self' URL for an embedding.

    Args:
        index_name: Name of the vector index (if known).
        embedding_id: UUID of the embedding.

    Returns:
        Relative URL path representing the embedding resource.
    """
    if index_name:
        return f"/vector/indices/{index_name}/embeddings/{embedding_id}"
    return f"/embeddings/{embedding_id}"


def build_index_self_url(index_name: str) -> str:
    """
    Build the 'self' URL for a vector index.

    Args:
        index_name: Name of the vector index.

    Returns:
        Relative URL path representing the vector index resource.
    """
    return f"/vector/indices/{index_name}"


def vector_index_to_model(index: VectorIndex) -> VectorIndexModel:
    """
    Convert a DB VectorIndex dataclass into a VectorIndexModel Pydantic schema.

    Args:
        index: Dataclass representing a vector_indices table row.

    Returns:
        VectorIndexModel suitable for API responses.
    """
    return VectorIndexModel(
        vector_index_name=index.vector_index_name,
        description=index.description,
        dimensions=index.dimensions,
        created_at=index.created_at,
        updated_at=index.updated_at,
        self=build_index_self_url(index.vector_index_name),
    )


def embedding_to_single_result(
    emb: Embedding,
    index: VectorIndex | None,
    include_index: bool,
    input_index: int | None = None,
) -> SingleEmbeddingResult | SingleEmbeddingResultNoIndex:
    """
    Convert a DB Embedding dataclass into an API embedding result.

    Args:
        emb: Dataclass representing an embeddings table row.
        index: Optional VectorIndex dataclass for the embedding.
        input_index: Position of this embedding in the original request/input.
        include_index: Whether to include index_name in the info block.

    Returns:
        Either SingleEmbeddingResult (with index_name) or
        SingleEmbeddingResultNoIndex depending on include_index and index.
    """
    vector = emb.embedding
    embedding_id = emb.embedding_id
    index_name = index.vector_index_name if index else None
    authz_url = f"/vector/indices/{index_name}" if index_name else "/vector/indices"

    if include_index and index:
        info = EmbeddingInfo(
            index_name=index_name,
            authz=authz_url,
            authz_version=emb.authz_version,
            self=build_embedding_self_url(index_name, embedding_id),
        )
        return SingleEmbeddingResult(
            vector=vector,
            input_index=input_index,
            embedding_id=embedding_id,
            info=info,
        )

    info_no_index = EmbeddingInfoNoIndex(
        authz=authz_url,
        authz_version=emb.authz_version,
        self=build_embedding_self_url(None, embedding_id),
    )
    return SingleEmbeddingResultNoIndex(
        vector=vector,
        input_index=input_index,
        embedding_id=embedding_id,
        info=info_no_index,
    )
