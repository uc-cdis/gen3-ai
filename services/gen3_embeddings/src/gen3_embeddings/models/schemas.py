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
"""
An authz resource path, checked against the policy engine and stored on the row.

Any resource path the caller holds the relevant action on, not just a collection's own
`/vectorstore/collections/{name}`. Length is the only constraint the service imposes; the
policy engine decides whether the path means anything.
"""

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

    collection_id: int = Field(..., alias="id", description="Numeric identifier for the collection.")
    collection_name: str = Field(..., description="The collection's normalized, lower-cased name.")
    description: str | None = Field(None, description="Free-text description supplied when the collection was created.")
    dimensions: int = Field(
        ...,
        description="Number of components in every vector in this collection. Fixed at creation.",
    )
    vector_type: VectorType = Field(
        ...,
        description=(
            "Storage precision for this collection's vectors: `vector` stores float32, `halfvec` "
            "stores float16. Fixed at creation."
        ),
    )
    created_at: datetime | None = Field(None, description="When the collection was created.")
    updated_at: datetime | None = Field(None, description="When the collection's metadata last changed.")
    self: str | None = Field(None, description="URL of this collection.")
    available_embeddings_count: int | None = Field(
        None,
        description=(
            "How many embeddings in this collection you can read. Only present when the request asked for `counts`."
        ),
    )


class PaginatedCollectionsResponse(BaseModel):
    """A page of collections the caller is authorized to see."""

    collections: list[CollectionModel] = Field(..., description="The collections on this page.")
    page: int = Field(..., description="The page number that was returned, counting from 1.")
    page_size: int = Field(..., description="The page size that was applied.")
    next_page: int | None = Field(
        None,
        description="Page number to request for more results, or absent when this is the last page.",
    )
    prev_page: int | None = Field(
        None, description="Page number of the previous page, or absent when this is the first page."
    )


class EmbeddingInfo(BaseModel):
    """Metadata about an embedding, omitted from responses when `exclude_info` is set."""

    collection_id: int = Field(..., description="Identifier of the collection holding this embedding.")
    authz: str = Field(
        ...,
        description="Authorization resource path this embedding is stored under.",
    )
    metadata: dict | None = Field(None, description="The arbitrary JSON metadata stored with this embedding.")
    self: str = Field(..., description="URL of this embedding.")


class SingleEmbeddingResult(BaseModel):
    """One embedding returned as a JSON array of floats."""

    vector: list[float] = Field(
        ...,
        description=(
            "The embedding, as a JSON array of floats. Values are read back at the collection's "
            "storage precision, so they may differ slightly from what was written."
        ),
    )
    input_index: int | None = Field(
        None,
        description=(
            "Zero-based position of the item in the request that produced this result, for lining "
            "results up with the order you sent. Absent on reads that had no input list."
        ),
    )
    embedding_id: UUID = Field(..., description="Identifier for this embedding.")
    info: EmbeddingInfo | None = Field(
        None, description="Where this embedding lives and what is stored with it. Absent when `exclude_info` is set."
    )


class SingleEmbeddingResultBinary(BaseModel):
    """
    One embedding returned as base64-encoded raw bytes.

    Cheaper than the JSON float array for large vectors. Decode according to `precision`.
    """

    vector_base64: bytes = Field(
        ...,
        description=(
            "The embedding as base64-encoded raw float bytes. Decode the base64, then read it as "
            "a little-endian array of floats of the width given by `precision`. The number of "
            "dimensions is the decoded byte length divided by that width."
        ),
    )
    precision: VectorPrecision = Field(
        ...,
        description=(
            "Width of each little-endian float in `vector_base64`: `float32` is 4 bytes, "
            "`float16` is 2. This is the collection's storage precision, not a conversion."
        ),
    )
    input_index: int | None = Field(
        None,
        description=(
            "Zero-based position of the requested UUID that produced this result. Absent when the "
            "request carried no input list."
        ),
    )
    embedding_id: UUID = Field(..., description="Identifier for this embedding.")
    info: EmbeddingInfo | None = Field(
        None, description="Where this embedding lives and what is stored with it. Absent when `exclude_info` is set."
    )

    model_config = ConfigDict(ser_json_bytes="base64")


class EmbeddingResponseWithCollections(BaseModel):
    """Embeddings plus the collections they came from, for cross-collection reads."""

    embeddings: list[SingleEmbeddingResult] = Field(..., description="The embeddings that were found.")
    collections: list[CollectionModel] | None = Field(
        None,
        description="Metadata for every collection represented in `embeddings`, so each result can be traced back.",
    )


class EmbeddingResponseBinaryWithCollections(BaseModel):
    """Binary embeddings plus the collections they came from, for cross-collection reads."""

    embeddings: list[SingleEmbeddingResultBinary] = Field(..., description="The embeddings that were found.")
    collections: list[CollectionModel] | None = Field(
        None,
        description="Metadata for every collection represented in `embeddings`, so each result can be traced back.",
    )


class EmbeddingResponse(BaseModel):
    """Embeddings from a single, already-known collection."""

    embeddings: list[SingleEmbeddingResult] = Field(
        ..., description="The embeddings, in the same order as the request that produced them."
    )


class EmbeddingResponseBinary(BaseModel):
    """Binary embeddings from a single, already-known collection."""

    embeddings: list[SingleEmbeddingResultBinary] = Field(..., description="The embeddings that were found.")
    count: int = Field(
        ...,
        description=(
            "How many embeddings are in `embeddings`. May be fewer than the number of UUIDs you "
            "asked for, since ones that do not resolve are omitted."
        ),
    )


class PaginatedEmbeddingResponse(BaseModel):
    """A page of embeddings from one collection."""

    embeddings: list[SingleEmbeddingResult] = Field(..., description="The embeddings on this page.")
    page: int = Field(..., description="The page number that was returned, counting from 1.")
    page_size: int = Field(..., description="The page size that was applied.")
    next_page: int | None = Field(
        None,
        description="Page number to request for more results, or absent when this is the last page.",
    )
    prev_page: int | None = Field(
        None, description="Page number of the previous page, or absent when this is the first page."
    )


class SearchRequestBody(BaseModel):
    """
    Request body for vector search operations.
    """

    input: Annotated[str, Field(max_length=MAX_TEXT_INPUT_LENGTH)] | Vector = Field(
        ...,
        description=(
            "The query vector, as an array of floats with the same number of dimensions as the "
            "collections being searched. A string is accepted by the schema so that text queries "
            "can be added later, but supplying one returns a 400 today."
        ),
    )
    top_k: TopK = Field(10, description="Maximum number of hits to return.")
    min_value: MetricBound | None = Field(
        None,
        description=(
            "Keep only hits whose `value` is greater than or equal to this. With a distance "
            "metric that means dropping hits that are too close; with `cosine_similarity` it "
            "means dropping hits that are not similar enough."
        ),
    )
    max_value: MetricBound | None = Field(
        None,
        description=(
            "Keep only hits whose `value` is less than or equal to this. With a distance metric "
            "that is the usual way to require a minimum closeness."
        ),
    )
    distance_metric: DistanceMetric = Field(
        DistanceMetric.cosine_similarity,
        description=(
            "How to score and order hits. All of these are distances, where smaller is closer, "
            "except `cosine_similarity`, where larger is closer."
        ),
    )
    filters: SearchFilters | None = Field(
        None,
        description=(
            "Restrict the search to embeddings whose metadata matches every one of these "
            "key/value pairs exactly. Values are compared as strings."
        ),
        examples=[{"source": "some_file.md"}],
    )


class SingleSearchResult(BaseModel):
    """
    Search result for a single hit.
    """

    id: UUID = Field(..., description="Identifier of the matching embedding.")
    distance_metric: DistanceMetric = Field(..., description="The metric `value` was computed with.")
    # distance or similarity depending on metric
    value: float = Field(
        ...,
        description=(
            "This hit's score under `distance_metric`. Smaller is closer for every metric except "
            "`cosine_similarity`, where larger is closer."
        ),
    )
    embedding: dict = Field(..., description="The matching embedding itself, in the same shape as a read result.")


class SearchResponse(BaseModel):
    """Ranked search hits, with the collections they were found in."""

    embeddings: list[SingleSearchResult] = Field(..., description="The hits, nearest first.")
    collections: list[CollectionModel] | None = Field(
        None, description="Metadata for the collections these hits came from."
    )


class CreateCollectionBody(BaseModel):
    """
    Request body for creating a new Collection.
    """

    collection_name: CollectionNameField = Field(
        ...,
        description=(
            "Name for the new collection. Lower-cased before it is stored, and limited to "
            "lowercase letters, digits, hyphen, and underscore."
        ),
        examples=["my-documents"],
    )
    description: DescriptionField | None = Field(None, description="Optional free-text description.")
    dimensions: Dimensions = Field(
        ...,
        description=(
            "Number of components every vector in this collection will have. Enforced on every "
            "write and cannot be changed later, so it must match the model you embed with."
        ),
        examples=[1536],
    )
    vector_type: VectorType = Field(
        VectorType.vector,
        description=(
            "Storage precision for this collection. `vector` stores float32; `halfvec` stores "
            "float16, which halves the storage at the cost of precision. Cannot be changed later."
        ),
    )


class UpdateCollectionBody(BaseModel):
    """
    Request body for updating mutable properties of a Collection.
    """

    description: DescriptionField | None = Field(
        None, description="Replacement free-text description for the collection."
    )


class UpdateEmbeddingBody(BaseModel):
    """
    Request body for updating an embedding.

    Every field is optional and only supplied ones are written. Setting `authz` moves the
    embedding to a different resource path, which you must hold `update` on.
    """

    embedding: Vector | None = Field(
        None,
        description=(
            "Replacement vector, which must match the collection's `dimensions`. Omit to leave the vector as it is."
        ),
    )
    metadata: Metadata | None = Field(
        None,
        description="Replacement metadata. Omit to leave the existing metadata as it is.",
        examples=[{"source": "some_file.md", "chunk_size": "1000"}],
    )
    authz: AuthzField | None = Field(
        None,
        description=(
            "Move the embedding to a different authorization resource path. You must hold "
            "`update` on the path you name. Omit to leave it where it is."
        ),
        examples=["/programs/my_program/projects/my_project"],
    )


class EmbeddingToCreate(BaseModel):
    """
    Data for creating a single embedding in a collection.

    'embedding' can be:
      - list[float] (already-embedded vector)
      - list[str]   (text chunks to be embedded later, not yet implemented here)

    For now, our code will only accept list[float] until we wire in the AI model
    service for text → embedding.
    """

    embedding: Vector | TextChunks = Field(
        ...,
        description=(
            "The vector to store, as an array of floats matching the collection's `dimensions`. "
            "An array of strings is accepted by the schema so that text input can be added later, "
            "but supplying one returns a 400 today."
        ),
    )
    metadata: Metadata | None = Field(
        None,
        description="Arbitrary JSON stored alongside the vector and returned with it. Searchable via `filters`.",
    )
    embedding_id: UUID | None = Field(
        None,
        description=(
            "On `PUT`, the id of an existing embedding to replace; an id that does not exist is "
            "rejected rather than created. Ignored on `POST`, where ids are always assigned by "
            "the service."
        ),
    )

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

    `authz` is applied to every embedding in the request, and may be any resource path you
    hold the action on -- e.g. "/programs/my_program/projects/my_project" or
    "/vectorstore/collections/my_collection". It defaults to the collection's own path when
    omitted, and a later PUT can change it.
    """

    authz: AuthzField | None = Field(
        None,
        description=(
            "Authorization resource path to store every embedding in this request under. You must "
            "hold the request's action on it. Defaults to the collection's own path, "
            "`/vectorstore/collections/{collection_name}`."
        ),
        examples=["/programs/my_program/projects/my_project"],
    )
    embeddings: Annotated[
        list[EmbeddingToCreate],
        # The per-item bounds above cap one embedding; this caps how many of them a single
        # request may carry, so the two multiply out to a worst case we can size a pod for.
        Field(
            min_length=1,
            max_length=MAX_EMBEDDINGS_PER_REQUEST,
            description=(
                "The embeddings to write. Results come back in this same order, each tagged with "
                f"its `input_index`. At most {MAX_EMBEDDINGS_PER_REQUEST} per request."
            ),
        ),
    ]
