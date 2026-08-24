"""Validation and conversion helpers between database rows and API schemas."""

import re
from uuid import UUID

from gen3_embeddings.config import MAX_COLLECTION_NAME_LENGTH
from gen3_embeddings.database.models import Collection, Embedding
from gen3_embeddings.models.schemas import (
    CollectionModel,
    EmbeddingInfo,
    SingleEmbeddingResult,
    SingleEmbeddingResultBinary,
    VectorPrecision,
)


def normalize_collection_name(name: str) -> str:
    """
    Normalize and validate a collection_name used in path or elsewhere.

    - strip whitespace
    - lower-case
    - bound the length, since this also runs on path parameters and query strings, which
      no Pydantic model gets to constrain first
    - ensure only [a-z0-9_-]

    Args:
        name (str): Raw collection name.

    Returns:
        str: The normalized name.

    Raises:
        ValueError: If the name is too long or contains characters outside [a-z0-9_-].
    """
    name = name.strip().lower()
    if len(name) > MAX_COLLECTION_NAME_LENGTH:
        raise ValueError(f"collection_name may be at most {MAX_COLLECTION_NAME_LENGTH} characters, got {len(name)}")
    pattern = re.compile(r"^[a-z0-9_-]+$")
    if not pattern.match(name):
        raise ValueError("collection_name may only contain lowercase letters, digits, hyphen (-), and underscore (_)")
    return name


def normalize_authz(authz: str | None) -> str | None:
    """
    Normalize and validate an authz string used in path or elsewhere.

    - strip whitespace
    - ensure it starts with a slash if not None or empty

    Args:
        authz (str | None): Raw authz resource path.

    Returns:
        str | None: The normalized path, or None if nothing was supplied.

    Raises:
        ValueError: If the path does not start with a slash.
    """
    if not authz:
        return None
    authz = authz.strip()
    if authz and not authz.startswith("/"):
        raise ValueError("authz must start with a slash (/)")
    return authz


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


def collection_to_model(col: Collection, available_embeddings_count: int | None = None) -> CollectionModel:
    """
    Convert a DB collection dataclass into a CollectionModel Pydantic schema.

    Args:
        col: Dataclass representing a collections table row.
        available_embeddings_count: Number of available embeddings in the collection.

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
        available_embeddings_count=available_embeddings_count,
    )


def embedding_to_binary_result(
    emb: Embedding,
    collection: Collection | None,
    exclude_info: bool = False,
    input_index: int | None = None,
    precision: VectorPrecision = VectorPrecision.float32,
) -> SingleEmbeddingResultBinary:
    """
    Convert a DB Embedding dataclass into an API embedding result.

    Args:
        emb: Dataclass representing an embeddings table row
        collection: Optional Collection dataclass for the embedding
        exclude_info: whether or not to exclude extra info per embedding
        input_index: Position of this embedding in the original request/input
        precision: string to represent prevision of the embedding for binary response

    Returns:
        SingleEmbeddingResultBinary object
    """
    info: EmbeddingInfo | None = None
    if not exclude_info:
        collection_name = None
        if collection:
            collection_name = collection.collection_name
        info = EmbeddingInfo(
            collection_id=emb.collection_id,
            authz=emb.authz,
            self=build_embedding_self_url(collection_name, emb.embedding_id),
            metadata=emb.metadata,
        )

    if hasattr(emb.embedding, "to_numpy"):
        # to_numpy() is a zero-copy native-byte-order view we can serialize directly.
        emb_bytes = emb.embedding.to_numpy().tobytes()
    else:
        # already a numpy array
        emb_bytes = emb.embedding.tobytes()

    return SingleEmbeddingResultBinary(
        vector_base64=emb_bytes,
        precision=precision,
        embedding_id=emb.embedding_id,
        input_index=input_index,
        info=info,
    )


def embedding_to_result(
    emb: Embedding,
    collection: Collection | None,
    exclude_info: bool = False,
    input_index: int | None = None,
) -> SingleEmbeddingResult:
    """
    Convert a DB Embedding dataclass into an API embedding result.

    Args:
        emb: Dataclass representing an embeddings table row.
        collection: Optional Collection dataclass for the embedding.
        exclude_info: whether or not to exclude extra info per embedding
        input_index: Position of this embedding in the original request/input.

    Returns:
        SingleEmbeddingResult object
    """
    info: EmbeddingInfo | None = None
    if not exclude_info:
        collection_name = None
        if collection:
            collection_name = collection.collection_name
        info = EmbeddingInfo(
            collection_id=emb.collection_id,
            authz=emb.authz,
            self=build_embedding_self_url(collection_name, emb.embedding_id),
            metadata=emb.metadata,
        )

    return SingleEmbeddingResult(
        vector=emb.embedding.to_list() if hasattr(emb.embedding, "to_list") else emb.embedding,
        embedding_id=emb.embedding_id,
        input_index=input_index,
        info=info,
    )
