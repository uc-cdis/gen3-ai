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

from fastapi import Body, Depends, HTTPException, Query

from gen3_embeddings.config import (
    MAX_AI_MODEL_NAME_LENGTH,
    MAX_COLLECTIONS_PER_SEARCH,
    MAX_COLLECTIONS_QUERY_LENGTH,
    MAX_EMBEDDING_UUIDS_PER_REQUEST,
    MAX_PAGE,
    MAX_PAGE_SIZE,
)
from gen3_embeddings.models.helpers import normalize_collection_name


def normalized_collection_name(collection_name: str) -> str:
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

Page = Annotated[int, Query(ge=1, le=MAX_PAGE)]
"""A 1-based page number. Bounded because it becomes an OFFSET, which Postgres takes as int4."""

PageSize = Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)]
"""Rows per page. Bounds the number of full vectors one response can carry."""

AiModel = Annotated[str | None, Query(max_length=MAX_AI_MODEL_NAME_LENGTH)]
"""The `ai_model` query parameter. Unused so far, but still caller-controlled free text."""

EmbeddingUUIDs = Annotated[
    list[UUID],
    Body(
        min_length=1,
        max_length=MAX_EMBEDDING_UUIDS_PER_REQUEST,
        examples=["embedding_uuid_0", "embedding_uuid_1"],
    ),
]
"""
A bulk-read UUID list.

The request itself is small at 36 bytes per UUID, but every entry that resolves returns a
full vector, so what this bounds is the response the caller can make us build.
"""
