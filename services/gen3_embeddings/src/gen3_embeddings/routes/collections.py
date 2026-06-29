from fastapi import APIRouter, Depends, HTTPException, Query, Request

from gen3_embeddings.auth import parse_and_auth_request
from gen3_embeddings.config import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE, logging
from gen3_embeddings.database.db import DataAccessLayer, get_data_access_layer
from gen3_embeddings.models.helpers import collection_to_model, normalize_collection_name
from gen3_embeddings.models.schemas import (
    CollectionModel,
    CreateCollectionBody,
    PaginatedCollectionsResponse,
    UpdateCollectionBody,
    VectorType,
)

collections_router = APIRouter()


@collections_router.get(
    "/vectorstore/collections",
    response_model=PaginatedCollectionsResponse,
    response_model_exclude_none=True,
    summary="Read all collections",
    tags=["Vectorstore Collections"],
)
@collections_router.get(
    "/vectorstore/collections/",
    include_in_schema=False,
)
async def list_collections(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    counts: bool = Query(False, alias="counts"),
    dal: DataAccessLayer = Depends(get_data_access_layer),
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
    collections = await dal.list_collections(offset=offset, limit=limit)

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


@collections_router.post(
    "/vectorstore/collections",
    response_model=CollectionModel,
    response_model_exclude_none=True,
    summary="Create collection",
    tags=["Vectorstore Collections"],
)
@collections_router.post(
    "/vectorstore/collections/",
    include_in_schema=False,
)
async def create_collection(
    request: Request,
    body: CreateCollectionBody,
    dal: DataAccessLayer = Depends(get_data_access_layer),
):
    """
    Create a new collection.

    If vector_type is not provided:
      - dimensions <= 2000     -> vector
      - 2000 < dimensions <= 4000 -> halfvec
      - dimensions > 4000      -> error (currently not supported)

    If vector_type IS provided:
      - vector   requires dimensions <= 2000
      - halfvec requires dimensions <= 4000
      - any dimensions > 4000          -> error

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

    # Decide vector_type if not provided
    if body.vector_type is None:
        if body.dimensions <= 2000:
            resolved_vector_type = VectorType.vector
        elif body.dimensions <= 4000:
            resolved_vector_type = VectorType.halfvec
        else:
            raise HTTPException(
                status_code=400,
                detail="Dimensions greater than 4000 are currently not supported",
            )
    else:
        resolved_vector_type = body.vector_type

        if body.dimensions > 4000:
            raise HTTPException(
                status_code=400,
                detail="Dimensions greater than 4000 are currently not supported",
            )
        if resolved_vector_type == VectorType.vector and body.dimensions > 2000:
            raise HTTPException(
                status_code=400,
                detail="For vector type 'vector', dimensions must be <= 2000",
            )
        if resolved_vector_type == VectorType.halfvec and body.dimensions > 4000:
            # This is redundant given the 4000 check above, but kept for clarity
            raise HTTPException(
                status_code=400,
                detail="For vector type 'halfvec', dimensions must be <= 4000",
            )

    col = await dal.create_collection(
        collection_name=normalized_name,
        description=body.description,
        dimensions=body.dimensions,
        vector_type=resolved_vector_type,
    )
    return collection_to_model(col)


@collections_router.get(
    "/vectorstore/collections/{collection_name}",
    response_model=CollectionModel,
    response_model_exclude_none=True,
    summary="Read collection info",
    tags=["Vectorstore Collections"],
    dependencies=[Depends(parse_and_auth_request)],
)
@collections_router.get(
    "/vectorstore/collections/{collection_name}/",
    include_in_schema=False,
    dependencies=[Depends(parse_and_auth_request)],
)
async def get_collection(
    collection_name: str,
    counts: bool = Query(False, alias="counts"),
    dal: DataAccessLayer = Depends(get_data_access_layer),
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

    col = await dal.get_collection_by_name(collection_name)

    if not col:
        raise HTTPException(status_code=404, detail="Collection not found")
    available_embeddings_count = None

    if counts:
        available_embeddings_count = await dal.count_available_embeddings_in_collection(col)
    return collection_to_model(col, available_embeddings_count=available_embeddings_count)


@collections_router.patch(
    "/vectorstore/collections/{collection_name}",
    summary="Update collection info",
    tags=["Vectorstore Collections"],
    response_model_exclude_none=True,
    dependencies=[Depends(parse_and_auth_request)],
)
@collections_router.patch(
    "/vectorstore/collections/{collection_name}/",
    include_in_schema=False,
    dependencies=[Depends(parse_and_auth_request)],
)
async def update_collection(
    collection_name: str,
    body: UpdateCollectionBody,
    dal: DataAccessLayer = Depends(get_data_access_layer),
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
    try:
        collection_name = normalize_collection_name(collection_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    col = await dal.update_collection(collection_name=collection_name, description=body.description)
    if not col:
        raise HTTPException(status_code=404, detail="Collection not found")

    return collection_to_model(col)


@collections_router.delete(
    "/vectorstore/collections/{collection_name}",
    status_code=204,
    summary="Delete collection",
    tags=["Vectorstore Collections"],
    dependencies=[Depends(parse_and_auth_request)],
)
@collections_router.delete(
    "/vectorstore/collections/{collection_name}/",
    include_in_schema=False,
    dependencies=[Depends(parse_and_auth_request)],
)
async def delete_collection(collection_name: str, dal: DataAccessLayer = Depends(get_data_access_layer)):
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
    try:
        collection_name = normalize_collection_name(collection_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    success = await dal.delete_collection(collection_name)
    logging.info(f"Deleted collection: {collection_name}.")
    if not success:
        raise HTTPException(status_code=404, detail="Collection not found")
    return None
