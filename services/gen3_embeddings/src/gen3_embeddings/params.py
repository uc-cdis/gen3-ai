"""
Reusable request parameter types for the HTTP boundary.

Canonicalizing untrusted input belongs at the edge, not in the data access layer. Doing it
here, as a type annotation, means every endpoint and every dependency that declares a
collection name gets the same normalized value without having to remember to call
`normalize_collection_name` itself.

That matters because the authorization check and the database lookup compare the name
against different things: `parse_and_auth_request` builds an Arborist resource path from it,
while the DAL compares it to the caller's allowed collection names. If only one side is
normalized, the two gates disagree - a mixed-case path could be denied by the policy engine
yet accepted by the DAL, or vice versa.

This module deliberately depends only on FastAPI and the model helpers, so both the routes
and `auth` can import it without a circular import.
"""

from typing import Annotated

from fastapi import Depends, HTTPException

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
"""A `collection_name` path parameter, normalized before any handler or dependency sees it."""
