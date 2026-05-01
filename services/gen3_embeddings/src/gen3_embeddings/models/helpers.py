from uuid import UUID

from gen3_embeddings.database.db import Collection, Embedding
from gen3_embeddings.models.schemas import (
    CollectionModel,
    EmbeddingInfo,
    SingleEmbeddingResult,
)


def build_embedding_self_url(collection_name: str | None, embedding_id: UUID) -> str:
    """
    Build the 'self' URL for an embedding.

    Args:
        collection_name: Name of the collection (if known).
        embedding_id: UUID of the embedding.

    Returns:
        Relative URL path representing the embedding resource.
    """
    if collection_name:
        return f"/vectorstore/collections/{collection_name}/embeddings/{embedding_id}"
    return f"/embeddings/{embedding_id}"


def build_collection_self_url(collection_name: str) -> str:
    """
    Build the 'self' URL for a collection.

    Args:
        collection_name: Name of the collection.

    Returns:
        Relative URL path representing the collection resource.
    """
    return f"/vectorstore/collections/{collection_name}"


def collection_to_model(col: Collection) -> CollectionModel:
    """
    Convert a DB collection dataclass into a CollectionModel Pydantic schema.

    Args:
        col: Dataclass representing a collections table row.

    Returns:
        CollectionModel suitable for API responses.
    """
    return CollectionModel(
        id=col.id,
        collection_name=col.collection_name,
        description=col.description,
        dimensions=col.dimensions,
        vector_type=col.vector_type,
        created_at=col.created_at,
        updated_at=col.updated_at,
        self=build_collection_self_url(col.collection_name),
    )


def embedding_to_result(
    emb: Embedding,
    collection: Collection | None,
    include_info: bool = True,
    input_index: int | None = None,
) -> SingleEmbeddingResult:
    """
    Convert a DB Embedding dataclass into an API embedding result.

    Args:
        emb: Dataclass representing an embeddings table row.
        collection: Optional Collection dataclass for the embedding.
        include_info: easily control `no_embeddings_info`.
        input_index: Position of this embedding in the original request/input.

    Returns:
        SingleEmbeddingResult object
    """
    info: EmbeddingInfo | None = None
    if include_info:
        collection_name = None
        if collection:
            collection_name = collection.collection_name
        info = EmbeddingInfo(
            collection_id=emb.collection_id,
            authz=emb.authz,
            authz_version=emb.authz_version,
            self=build_embedding_self_url(collection_name, emb.embedding_id),
            metadata=emb.metadata,
        )

    return SingleEmbeddingResult(
        vector=emb.embedding,
        embedding_id=emb.embedding_id,
        input_index=input_index,
        info=info,
    )
