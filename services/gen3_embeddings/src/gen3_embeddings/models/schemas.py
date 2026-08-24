"""Pydantic request and response schemas for the public API."""

from datetime import datetime
from enum import StrEnum
from typing import Annotated
from uuid import UUID

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from gen3_embeddings.config import (
    MAX_AUTHZ_LENGTH,
    MAX_COLLECTION_NAME_LENGTH,
    MAX_DESCRIPTION_LENGTH,
    MAX_EMBEDDINGS_PER_REQUEST,
    MAX_SEARCH_FILTER_KEY_LENGTH,
    MAX_SEARCH_FILTER_VALUE_LENGTH,
    MAX_SEARCH_FILTERS,
    MAX_TEXT_CHUNKS,
    MAX_TEXT_INPUT_LENGTH,
    MAX_TOP_K,
    MAX_VECTOR_DIMENSIONS,
    MIN_VECTOR_DIMENSIONS,
)
from gen3_embeddings.limits import validate_metadata

# Bounded aliases for the caller-supplied fields whose cost to this service scales with
# their size. They are declared once here so every request schema that accepts one gets the
# same ceiling, and so the OpenAPI document advertises the limits instead of leaving callers
# to discover them by getting a 422.
#
# These are a denial-of-service control: see the request limits block in `config` for the
# reasoning behind each number, and `limits` for the byte ceiling that backs them all up.

VectorComponent = Annotated[float, Field(allow_inf_nan=False)]
"""One element of a vector. pgvector rejects NaN and infinity, so refuse them at the edge."""

Vector = Annotated[
    list[VectorComponent],
    Field(min_length=MIN_VECTOR_DIMENSIONS, max_length=MAX_VECTOR_DIMENSIONS),
]
"""A vector of floats, bounded independently of any collection's `dimensions`."""

TextChunks = Annotated[
    list[Annotated[str, Field(max_length=MAX_TEXT_INPUT_LENGTH)]],
    Field(max_length=MAX_TEXT_CHUNKS),
]
"""Raw text to be embedded. Not implemented yet, but still bounded."""

CollectionNameField = Annotated[str, Field(min_length=1, max_length=MAX_COLLECTION_NAME_LENGTH)]
"""A collection name in a request body. Path parameters are bounded in `models.helpers`."""

DescriptionField = Annotated[str, Field(max_length=MAX_DESCRIPTION_LENGTH)]
"""Free text stored on a collection."""

AuthzField = Annotated[str, Field(max_length=MAX_AUTHZ_LENGTH)]
"""An authz resource path, which is sent to the policy engine and stored on every row."""

Metadata = Annotated[dict, AfterValidator(validate_metadata)]
"""Embedding metadata, bounded by serialized size, top-level key count, and nesting depth."""

SearchFilters = Annotated[
    dict[
        Annotated[str, Field(max_length=MAX_SEARCH_FILTER_KEY_LENGTH)],
        Annotated[str, Field(max_length=MAX_SEARCH_FILTER_VALUE_LENGTH)],
    ],
    Field(max_length=MAX_SEARCH_FILTERS),
]
"""Metadata equality filters. Each one adds a WHERE clause and two parameters to the query."""

Dimensions = Annotated[int, Field(ge=MIN_VECTOR_DIMENSIONS, le=MAX_VECTOR_DIMENSIONS)]
"""A collection's vector width, which every embedding written to it is checked against."""

TopK = Annotated[int, Field(ge=1, le=MAX_TOP_K)]
"""Number of search hits to return. Becomes a SQL LIMIT over rows that each hold a vector."""

MetricBound = Annotated[float, Field(allow_inf_nan=False)]
"""A min/max threshold on a distance metric, compared in SQL."""


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

    input: Annotated[str, Field(max_length=MAX_TEXT_INPUT_LENGTH)] | Vector
    top_k: TopK = 10
    min_value: MetricBound | None = None
    max_value: MetricBound | None = None
    distance_metric: DistanceMetric = DistanceMetric.cosine_similarity
    filters: SearchFilters | None = None


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

    collection_name: CollectionNameField
    description: DescriptionField | None = None
    dimensions: Dimensions
    vector_type: VectorType = VectorType.vector


class UpdateCollectionBody(BaseModel):
    """
    Request body for updating mutable properties of a Collection.
    """

    description: DescriptionField | None = None


class UpdateEmbeddingBody(BaseModel):
    """
    Request body for updating an embedding.
    """

    embedding: Vector | None = None
    metadata: Metadata | None = None
    authz: AuthzField | None = None


class EmbeddingToCreate(BaseModel):
    """
    Data for creating a single embedding in a collection.

    'embedding' can be:
      - list[float] (already-embedded vector)
      - list[str]   (text chunks to be embedded later, not yet implemented here)

    For now, our code will only accept list[float] until we wire in the AI model
    service for text → embedding.
    """

    embedding: Vector | TextChunks
    metadata: Metadata | None = None
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

    authz: AuthzField | None = None
    embeddings: Annotated[
        list[EmbeddingToCreate],
        # The per-item bounds above cap one embedding; this caps how many of them a single
        # request may carry, so the two multiply out to a worst case we can size a pod for.
        Field(min_length=1, max_length=MAX_EMBEDDINGS_PER_REQUEST),
    ]
