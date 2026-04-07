from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class VectorIndexModel(BaseModel):
    """
    API schema representing a vector index.
    """

    index_name: str = Field(..., alias="vector_index_name")
    description: str | None = None
    dimensions: int
    created: datetime | None = Field(None, alias="created_at")
    updated: datetime | None = Field(None, alias="updated_at")
    self: str | None = None

    model_config = ConfigDict(
        validate_by_name=True,
    )


class VectorIndiceNamesModel(BaseModel):
    """
    API schema representing a collection of vector indices.
    """

    vector_indices: list[VectorIndexModel]


class EmbeddingInfoNoIndex(BaseModel):
    """
    Info block for an embedding when index metadata is not returned.
    """

    authz: str
    authz_version: int
    self: str


class EmbeddingInfo(BaseModel):
    """
    Info block for an embedding when index metadata is included.
    """

    index_name: str
    authz: str
    authz_version: int
    self: str


class SingleEmbeddingResultNoIndex(BaseModel):
    """
    Result schema for a single embedding without explicit index_name.
    """

    vector: list[float]
    input_index: int | None = None
    embedding_id: UUID
    info: EmbeddingInfoNoIndex | None = None


class SingleEmbeddingResult(BaseModel):
    """
    Result schema for a single embedding with index_name in the info block.
    """

    vector: list[float]
    input_index: int | None = None
    embedding_id: UUID
    info: EmbeddingInfo | None = None


class EmbeddingResponse(BaseModel):
    """
    Response schema for embedding operations that include index metadata.
    """

    embeddings: list[SingleEmbeddingResult]
    vector_indices: list[VectorIndexModel] | None = None


class EmbeddingResponseNoIndex(BaseModel):
    """
    Response schema for embedding operations that do not include index metadata.
    """

    embeddings: list[SingleEmbeddingResultNoIndex]


class SearchRequestBody(BaseModel):
    """
    Request body for vector search operations.
    """

    input: str | list[float]
    top_k: int = 10
    range: float | None = Field(None, alias="range")
    filters: dict[str, str] | None = None


class SingleSearchResultNoIndex(BaseModel):
    """
    Search result for a single hit when index metadata is not included.
    """

    id: UUID
    score: float = Field(..., alias="similarity_score")
    embedding: SingleEmbeddingResultNoIndex


class SingleSearchResult(BaseModel):
    """
    Search result for a single hit when index metadata is included.
    """

    id: UUID
    score: float = Field(..., alias="similarity_score")
    embedding: SingleEmbeddingResult


class SearchResponseNoIndex(BaseModel):
    """
    Response schema for searches scoped to a single index.
    """

    embeddings: list[SingleSearchResultNoIndex]
    vector_indices: list[VectorIndexModel] | None = None


class SearchResponse(BaseModel):
    """
    Response schema for searches across one or more indices.
    """

    embeddings: list[SingleSearchResult]


class CreateIndexBody(BaseModel):
    """
    Request body for creating a new vector index.
    """

    index_name: str
    description: str | None = None
    dimensions: int


class UpdateIndexBody(BaseModel):
    """
    Request body for updating mutable properties of a vector index.
    """

    description: str | None = None


class UpdateEmbeddingBody(BaseModel):
    """
    Request body for updating an embedding in an index.
    """

    embedding: list[float]
