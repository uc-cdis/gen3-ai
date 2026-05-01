from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field


class VectorType(StrEnum):
    vector = "vector"
    halfvec = "halfvec"


class DistanceMetric(StrEnum):
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


class PaginatedCollectionsResponse(BaseModel):
    collections: list[CollectionModel]
    page: int
    page_size: int
    next_page: int | None = None
    prev_page: int | None = None


class EmbeddingInfo(BaseModel):
    collection_id: int
    authz: list[str]
    authz_version: int
    metadata: dict | None = None
    self: str


class SingleEmbeddingResult(BaseModel):
    vector: list[float]
    input_index: int | None = None
    embedding_id: UUID
    info: EmbeddingInfo | None = None


class EmbeddingResponseWithCollections(BaseModel):
    embeddings: list[dict]
    collections: list[CollectionModel] | None = None


class EmbeddingResponse(BaseModel):
    embeddings: list[dict]


class PaginatedEmbeddingResponse(BaseModel):
    embeddings: list[dict]
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

    model_config = {
        "json_schema_extra": {
            "example": {
                "embedding": [0.1, 0.2, 0.3],
                "metadata": {
                    "source": "some_file.md",
                    "chunk_size": "1000",
                },
            }
        }
    }


class CreateEmbeddingsBody(BaseModel):
    """
    Data for creating embeddings in a collection.
    """

    embeddings: list[EmbeddingToCreate]
