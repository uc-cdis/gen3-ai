"""Pydantic request and response schemas for the public API."""

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class VectorPrecision(StrEnum):
    """Floating-point precision of the vectors in a binary embedding response."""

    float16 = "float16"
    float32 = "float32"


class VectorType(StrEnum):
    """pgvector column type backing a collection, which fixes its storage precision."""

    vector = "vector"
    halfvec = "halfvec"

    @property
    def precision(self) -> VectorPrecision:
        """
        Return the precision that vectors of this type are stored at.

        Returns:
            VectorPrecision: float32 for `vector`, float16 for `halfvec`.

        Raises:
            ValueError: If the enum gains a member without a mapped precision.
        """
        if self == VectorType.vector:
            return VectorPrecision.float32
        if self == VectorType.halfvec:
            return VectorPrecision.float16
        raise ValueError(f"Unsupported vector type: {self}")


class DistanceMetric(StrEnum):
    """
    Metric used to rank search results.

    All but `cosine_similarity` are distances, where smaller is closer.
    """

    l2_distance = "l2_distance"
    inner_product = "inner_product"
    cosine_distance = "cosine_distance"
    l1_distance = "l1_distance"
    cosine_similarity = "cosine_similarity"


class CollectionModel(BaseModel):
    """
    API schema representing a collection.
    """

    collection_id: int = Field(..., alias="id")
    collection_name: str
    description: str | None = None
    dimensions: int
    vector_type: VectorType
    created_at: datetime | None = None
    updated_at: datetime | None = None
    self: str | None = None
    available_embeddings_count: int | None = None


class PaginatedCollectionsResponse(BaseModel):
    """A page of collections the caller is authorized to see."""

    collections: list[CollectionModel]
    page: int
    page_size: int
    next_page: int | None = None
    prev_page: int | None = None


class EmbeddingInfo(BaseModel):
    """Metadata about an embedding, omitted from responses when `exclude_info` is set."""

    collection_id: int
    authz: str
    metadata: dict | None = None
    self: str


class SingleEmbeddingResult(BaseModel):
    """One embedding returned as a JSON array of floats."""

    vector: list[float]  # TODO: try to switch to Vector / HalfVector
    input_index: int | None = None
    embedding_id: UUID
    info: EmbeddingInfo | None = None

    # this is to support Vector / HalfVector
    model_config = ConfigDict(arbitrary_types_allowed=True)


class SingleEmbeddingResultBinary(BaseModel):
    """
    One embedding returned as base64-encoded raw bytes.

    Cheaper than the JSON float array for large vectors. Decode according to `precision`.
    """

    vector_base64: bytes
    precision: VectorPrecision
    input_index: int | None = None
    embedding_id: UUID
    info: EmbeddingInfo | None = None

    model_config = ConfigDict(ser_json_bytes="base64")


class EmbeddingResponseWithCollections(BaseModel):
    """Embeddings plus the collections they came from, for cross-collection reads."""

    embeddings: list[SingleEmbeddingResult]
    collections: list[CollectionModel] | None = None


class EmbeddingResponseBinaryWithCollections(BaseModel):
    """Binary embeddings plus the collections they came from, for cross-collection reads."""

    embeddings: list[SingleEmbeddingResultBinary]
    collections: list[CollectionModel] | None = None


class EmbeddingResponse(BaseModel):
    """Embeddings from a single, already-known collection."""

    embeddings: list[SingleEmbeddingResult]


class EmbeddingResponseBinary(BaseModel):
    """Binary embeddings from a single, already-known collection."""

    embeddings: list[SingleEmbeddingResultBinary]
    count: int


class PaginatedEmbeddingResponse(BaseModel):
    """A page of embeddings from one collection."""

    embeddings: list[SingleEmbeddingResult]
    page: int
    page_size: int
    next_page: int | None = None
    prev_page: int | None = None


class SearchRequestBody(BaseModel):
    """
    Request body for vector search operations.
    """

    input: str | list[float]
    top_k: int = 10
    min_value: float | None = None
    max_value: float | None = None
    distance_metric: DistanceMetric = DistanceMetric.cosine_similarity
    filters: dict[str, str] | None = None


class SingleSearchResult(BaseModel):
    """
    Search result for a single hit.
    """

    id: UUID
    distance_metric: DistanceMetric
    # distance or similarity depending on metric
    value: float
    embedding: dict


class SearchResponse(BaseModel):
    """Ranked search hits, with the collections they were found in."""

    embeddings: list[SingleSearchResult]
    collections: list[CollectionModel] | None = None


class CreateCollectionBody(BaseModel):
    """
    Request body for creating a new Collection.
    """

    collection_name: str
    description: str | None = None
    dimensions: int
    vector_type: VectorType = VectorType.vector


class UpdateCollectionBody(BaseModel):
    """
    Request body for updating mutable properties of a Collection.
    """

    description: str | None = None


class UpdateEmbeddingBody(BaseModel):
    """
    Request body for updating an embedding.
    """

    embedding: list[float] | None = None
    metadata: dict | None = None
    authz: str | None = None


class EmbeddingToCreate(BaseModel):
    """
    Data for creating a single embedding in a collection.

    'embedding' can be:
      - list[float] (already-embedded vector)
      - list[str]   (text chunks to be embedded later, not yet implemented here)

    For now, our code will only accept list[float] until we wire in the AI model
    service for text → embedding.
    """

    embedding: list[float] | list[str]
    metadata: dict | None = None
    embedding_id: UUID | None = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "embedding": [0.1, 0.2, 0.3],
                "metadata": {
                    "source": "some_file.md",
                    "chunk_size": "1000",
                },
                "embedding_id": "00000000-0000-0000-0000-000000000000",
            }
        }
    }


class CreateEmbeddingsBody(BaseModel):
    """
    Data for creating embeddings in a collection.
    authz example: "authz": "/vectorstore/collections/my_collection"
    """

    authz: str | None = None
    embeddings: list[EmbeddingToCreate]
