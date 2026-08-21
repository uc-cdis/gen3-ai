"""Routes for creating, reading, updating, and deleting vector collections."""

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from starlette import status

from common.fastapi.responses import (
    AUTH_RESPONSES,
    BAD_REQUEST_RESPONSE,
    NO_CONTENT_RESPONSE,
    not_found_response,
)
from gen3_embeddings.auth import parse_and_auth_request
from gen3_embeddings.config import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE, logging
from gen3_embeddings.database.db import DataAccessLayer
from gen3_embeddings.dependencies import get_allowed_collection_names, get_data_access_layer
from gen3_embeddings.models.helpers import collection_to_model, normalize_collection_name
from gen3_embeddings.models.schemas import (
    CollectionModel,
    CreateCollectionBody,
    PaginatedCollectionsResponse,
    UpdateCollectionBody,
)
from gen3_embeddings.params import CollectionName
from gen3_embeddings.routes.helpers import dual_path

collections_router = APIRouter()


@dual_path(
    collections_router,
    "get",
    "/vectorstore/collections",
    response_model=PaginatedCollectionsResponse,
    response_model_exclude_none=True,
    summary="Read all collections",
    description=(
        "Returns every vector collection you have access to, one page at a time. "
        "Set `counts=true` to also get the number of embeddings available in each collection."
    ),
    responses={**AUTH_RESPONSES},
    tags=["Vectorstore Collections"],
)
async def list_collections(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    counts: bool = Query(False, alias="counts"),
    dal: DataAccessLayer = Depends(get_data_access_layer),
    allowed_collection_names: set[str] = Depends(get_allowed_collection_names),
):
    """
    List all existing collections.

    Args:
        request: The request object
        dal: Data access layer dependency.
        counts: Whether to include available embeddings count.

    Returns:
        PaginatedCollectionsResponse containing all collections.
    """
    offset = (page - 1) * page_size
    limit = page_size

    logging.debug(f"Listing collections, offset={offset}, limit={limit}")
    collections = await dal.list_collections(
        allowed_collection_names=allowed_collection_names, offset=offset, limit=limit
    )

    # If counts=true, compute per-collection embedding counts
    counts_by_id: dict[int, int] = {}
    if counts:
        for col in collections:
            counts_by_id[col.id] = await dal.count_available_embeddings_in_collection(collection=col)

    next_page = page + 1 if len(collections) == page_size else None
    prev_page = page - 1 if page > 1 else None

    return PaginatedCollectionsResponse(
        collections=[
            collection_to_model(col, available_embeddings_count=counts_by_id.get(col.id)) for col in collections
        ],
        page=page,
        page_size=page_size,
        next_page=next_page,
        prev_page=prev_page,
    )


@dual_path(
    collections_router,
    "post",
    "/vectorstore/collections",
    response_model=CollectionModel,
    response_model_exclude_none=True,
    summary="Create collection",
    description=(
        "Creates a new vector collection. The collection name is normalized before it is stored, "
        "and the dimensions you supply here are enforced on every embedding added to the collection."
    ),
    responses={**AUTH_RESPONSES, **BAD_REQUEST_RESPONSE},
    tags=["Vectorstore Collections"],
)
async def create_collection(
    request: Request,
    body: CreateCollectionBody,
    dal: DataAccessLayer = Depends(get_data_access_layer),
    allowed_collection_names: set[str] = Depends(get_allowed_collection_names),
):
    """
    Create a new collection.

    Args:
        request: The request object
        body: Request body containing collection_name, description, and dimensions.
        dal: Data access layer dependency.

    Returns:
        CollectionModel for the created collection.
    """
    logging.debug(f"Creating collection: {body}...")

    try:
        normalized_name = normalize_collection_name(body.collection_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    col = await dal.create_collection(
        allowed_collection_names=allowed_collection_names,
        collection_name=normalized_name,
        description=body.description,
        dimensions=body.dimensions,
        vector_type=body.vector_type,
    )
    return collection_to_model(col)


@dual_path(
    collections_router,
    "get",
    "/vectorstore/collections/{collection_name}",
    response_model=CollectionModel,
    response_model_exclude_none=True,
    summary="Read collection info",
    description=(
        "Returns the metadata for a single collection, including its dimensions and vector type. "
        "Set `counts=true` to also get the number of embeddings available in it."
    ),
    responses={
        **AUTH_RESPONSES,
        **not_found_response("Collection"),
    },
    tags=["Vectorstore Collections"],
    dependencies=[Depends(parse_and_auth_request)],
)
async def get_collection(
    collection_name: CollectionName,
    counts: bool = Query(False, alias="counts"),
    dal: DataAccessLayer = Depends(get_data_access_layer),
    allowed_collection_names: set[str] = Depends(get_allowed_collection_names),
):
    """
    Read information about a specific collection.

    Args:
        collection_name: Name of the collection.
        dal: Data access layer dependency.

    Returns:
        CollectionModel representing the collection.

    Raises:
        HTTPException: 404 if collection is not found.
    """
    logging.debug(f"Getting collection: {collection_name}...")

    col = await dal.get_collection_by_name(collection_name, allowed_collection_names=allowed_collection_names)

    if not col:
        raise HTTPException(status_code=404, detail="Collection not found")
    available_embeddings_count = None

    if counts:
        available_embeddings_count = await dal.count_available_embeddings_in_collection(col)
    return collection_to_model(col, available_embeddings_count=available_embeddings_count)


@dual_path(
    collections_router,
    "patch",
    "/vectorstore/collections/{collection_name}",
    summary="Update collection info",
    description=(
        "Updates the mutable metadata on a collection, such as its description. "
        "The collection's name, dimensions, and vector type cannot be changed after creation."
    ),
    responses={
        **AUTH_RESPONSES,
        **BAD_REQUEST_RESPONSE,
        **not_found_response("Collection"),
    },
    tags=["Vectorstore Collections"],
    response_model_exclude_none=True,
    dependencies=[Depends(parse_and_auth_request)],
)
async def update_collection(
    collection_name: CollectionName,
    body: UpdateCollectionBody,
    dal: DataAccessLayer = Depends(get_data_access_layer),
    allowed_collection_names: set[str] = Depends(get_allowed_collection_names),
):
    """
    Update mutable metadata fields for a collection.

    Args:
        collection_name: Name of the collection to update.
        body: Request body containing fields to update (e.g., description).
        dal: Data access layer dependency.

    Returns:
        A simple success status dict.

    Raises:
        HTTPException: 404 if collection is not found.
    """
    logging.debug(f"Updating collection: {collection_name} with description={body.description}...")
    col = await dal.update_collection(
        collection_name=collection_name, description=body.description, allowed_collection_names=allowed_collection_names
    )
    if not col:
        raise HTTPException(status_code=404, detail="Collection not found")

    return collection_to_model(col)


@dual_path(
    collections_router,
    "delete",
    "/vectorstore/collections/{collection_name}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete collection",
    description=("Permanently deletes a collection and every embedding stored in it. This cannot be undone."),
    responses={
        **NO_CONTENT_RESPONSE,
        **AUTH_RESPONSES,
        **BAD_REQUEST_RESPONSE,
        **not_found_response("Collection"),
    },
    tags=["Vectorstore Collections"],
    dependencies=[Depends(parse_and_auth_request)],
)
async def delete_collection(
    collection_name: CollectionName,
    dal: DataAccessLayer = Depends(get_data_access_layer),
    allowed_collection_names: set[str] = Depends(get_allowed_collection_names),
):
    """
    Delete a collection by name.

    Args:
        collection_name: Name of the collection to delete.
        dal: Data access layer dependency.

    Returns:
        None on success.

    Raises:
        HTTPException: 404 if collection is not found.
    """
    success = await dal.delete_collection(collection_name, allowed_collection_names=allowed_collection_names)
    logging.info(f"Deleted collection: {collection_name}.")
    if not success:
        raise HTTPException(status_code=404, detail="Collection not found")
    return None
