"""
Reusable request parameter types for the HTTP boundary.

Canonicalizing untrusted input belongs at the edge, not in the data access layer. Doing it
here, as a type annotation, means every endpoint and every dependency that declares a
collection name gets the same normalized value without having to remember to call
`normalize_collection_name` itself.

That matters because the authorization check and the database lookup compare the name
against different things: `dependencies.authz` builds an Arborist resource path from it,
while the database compares it to the caller's allowed collection names. If only one side is
normalized, the two gates disagree - a mixed-case path could be denied by the policy engine
yet accepted by row-level security, or vice versa.

This module deliberately depends only on FastAPI and the model helpers, so both the routes
and `dependencies` can import it without a circular import.

It is also where the caller-controlled path, query, and body parameters get their upper
bounds. The equivalents for request *bodies* live on the schemas in `models.schemas`, but a
path or query parameter has no schema to carry a constraint, so it is declared here instead.
See the request limits block in `config` for why each bound exists.
"""

from typing import Annotated
from uuid import UUID

from fastapi import Body, Depends, HTTPException, Path, Query

from gen3_embeddings.config import (
    MAX_AI_MODEL_NAME_LENGTH,
    MAX_COLLECTION_NAME_LENGTH,
    MAX_COLLECTIONS_PER_SEARCH,
    MAX_COLLECTIONS_QUERY_LENGTH,
    MAX_EMBEDDING_UUIDS_PER_REQUEST,
    MAX_PAGE,
    MAX_PAGE_SIZE,
)
from gen3_embeddings.models.helpers import normalize_collection_name


def normalized_collection_name(
    collection_name: Annotated[
        str,
        Path(
            description=(
                "Name of the collection. Matched case-insensitively -- names are lower-cased "
                "before lookup, so `MyDocs` and `mydocs` are the same collection. May contain "
                f"only lowercase letters, digits, hyphen, and underscore, up to "
                f"{MAX_COLLECTION_NAME_LENGTH} characters; anything else returns a 400."
            ),
            examples=["my-documents"],
        ),
    ],
) -> str:
    """
    Normalize a `collection_name` request parameter, rejecting invalid ones.

    Args:
        collection_name (str): Raw value as supplied by the caller.

    Returns:
        str: The normalized collection name.

    Raises:
        HTTPException: 400 if the name is not a valid collection name.
    """
    try:
        return normalize_collection_name(collection_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


CollectionName = Annotated[str, Depends(normalized_collection_name)]
"""A `collection_name` path parameter, normalized before any handler or dependency sees it.

`normalize_collection_name` also bounds the length, which is the only bound this parameter
gets: a path segment is not covered by any request schema.
"""


def requested_collection_names(
    collections: Annotated[
        str | None,
        Query(
            max_length=MAX_COLLECTIONS_QUERY_LENGTH,
            description="Comma-separated collection names to restrict the search to.",
        ),
    ] = None,
) -> list[str] | None:
    """
    Split the comma-separated `collections` query parameter into a bounded list of names.

    The parameter is bounded twice, because each bound catches a different cost. The
    `max_length` above caps the string before it is split, since the split is what allocates.
    The count check below caps the names that come out of it, because the route makes one
    database round trip per name, so the work scales with how many were listed rather than
    with how long the string was.

    Args:
        collections (str | None): Raw comma-separated value, or None if not supplied.

    Returns:
        list[str] | None: The listed names, or None if the caller did not restrict the
        search. An empty list means the caller supplied only separators, which restricts
        the search to nothing.

    Raises:
        HTTPException: 400 if more collections were listed than the configured maximum.
    """
    if not collections:
        return None

    names = [value.strip() for value in collections.split(",") if value.strip()]
    if len(names) > MAX_COLLECTIONS_PER_SEARCH:
        raise HTTPException(
            status_code=400,
            detail=(
                f"At most {MAX_COLLECTIONS_PER_SEARCH} collections may be searched at once, "
                f"got {len(names)}. Split the search into smaller requests."
            ),
        )
    return names


RequestedCollectionNames = Annotated[list[str] | None, Depends(requested_collection_names)]
"""The `collections` query parameter on cross-collection search, split and bounded."""

Page = Annotated[
    int,
    Query(
        ge=1,
        le=MAX_PAGE,
        description=(
            "Which page of results to return, counting from 1. Use the `next_page` and "
            "`prev_page` values in the response to walk through pages rather than computing "
            f"them yourself. Maximum {MAX_PAGE}."
        ),
    ),
]
"""A 1-based page number. Bounded because it becomes an OFFSET, which Postgres takes as int4."""

PageSize = Annotated[
    int,
    Query(
        ge=1,
        le=MAX_PAGE_SIZE,
        description=(
            "How many results to return per page. Each result carries a full vector, so large "
            f"pages mean large responses. Maximum {MAX_PAGE_SIZE}."
        ),
    ),
]
"""Rows per page. Bounds the number of full vectors one response can carry."""

AiModel = Annotated[
    str | None,
    Query(
        max_length=MAX_AI_MODEL_NAME_LENGTH,
        description=(
            "Reserved for selecting the model used to embed text input. Not implemented yet: "
            "supplying it has no effect, because only pre-computed vectors are accepted today."
        ),
    ),
]
"""The `ai_model` query parameter. Unused so far, but still caller-controlled free text."""

ExcludeInfo = Annotated[
    bool,
    Query(
        description=(
            "Omit the per-embedding `info` object (collection, authz, metadata, and self link) "
            "from the response, leaving just the vectors and their IDs. Useful when you already "
            "know where the embeddings came from and want a smaller response."
        ),
    ),
]
"""Whether to drop the `info` block from each returned embedding."""

Counts = Annotated[
    bool,
    Query(
        description=(
            "Include `available_embeddings_count` on each collection. Off by default because it "
            "costs an extra count query per collection returned."
        ),
    ),
]
"""Whether to count each collection's embeddings, which is an extra query per collection."""

EmbeddingUUID = Annotated[
    UUID,
    Path(
        description="ID of the embedding, as returned in `embedding_id` when it was created.",
        examples=["00000000-0000-0000-0000-000000000000"],
    ),
]
"""An `embedding_uuid` path parameter."""

EmbeddingUUIDs = Annotated[
    list[UUID],
    Body(
        min_length=1,
        max_length=MAX_EMBEDDING_UUIDS_PER_REQUEST,
        description=(
            "The IDs of the embeddings to read, as a JSON array. Results come back in the same "
            f"order you asked for them. At most {MAX_EMBEDDING_UUIDS_PER_REQUEST} per request."
        ),
        # A list of one example VALUE, not a list of example items: the example for a list
        # parameter has to itself be a list, or Redoc renders a bare string in place of the array.
        examples=[["00000000-0000-0000-0000-000000000000", "11111111-1111-1111-1111-111111111111"]],
    ),
]
"""
A bulk-read UUID list.

The request itself is small at 36 bytes per UUID, but every entry that resolves returns a
full vector, so what this bounds is the response the caller can make us build.
"""
