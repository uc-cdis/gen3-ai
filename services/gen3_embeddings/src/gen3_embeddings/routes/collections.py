from fastapi import APIRouter, Depends, HTTPException, Query, Request

from gen3_embeddings.auth import parse_and_auth_request
from gen3_embeddings.config import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE
from gen3_embeddings.db import DataAccessLayer, get_data_access_layer
from gen3_embeddings.models.helpers import collection_to_model
from gen3_embeddings.models.schemas import (
    CollectionModel,
    CreateCollectionBody,
    PaginatedCollectionsResponse,
    UpdateCollectionBody,
)

collections_router = APIRouter(tags=["Vectorstore Collections"])


@collections_router.get(
    "/vectorstore/collections",
    response_model=PaginatedCollectionsResponse,
    summary="Read all collections",
)
async def list_collections(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    dal: DataAccessLayer = Depends(get_data_access_layer),
):
    """
    List all existing collections.

    Args:
        request: The request object
        dal: Data access layer dependency.

    Returns:
        PaginatedCollectionsResponse containing all collections.
    """
    await parse_and_auth_request(request, "")
    offset = (page - 1) * page_size
    limit = page_size

    collections = await dal.list_collections(offset=offset, limit=limit)

    next_page = page + 1 if len(collections) == page_size else None
    prev_page = page - 1 if page > 1 else None

    return PaginatedCollectionsResponse(
        collections=[collection_to_model(col) for col in collections],
        page=page,
        page_size=page_size,
        next_page=next_page,
        prev_page=prev_page,
    )


@collections_router.post(
    "/vectorstore/collections",
    response_model=CollectionModel,
    summary="Create collection",
)
async def create_collection(
    request: Request,
    body: CreateCollectionBody,
    dal: DataAccessLayer = Depends(get_data_access_layer),
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
    await parse_and_auth_request(request, body.collection_name)
    col = await dal.create_collection(
        collection_name=body.collection_name,
        description=body.description,
        dimensions=body.dimensions,
    )
    return collection_to_model(col)


@collections_router.get(
    "/vectorstore/collections/{collection_name}",
    response_model=CollectionModel,
    summary="Read collection info",
    dependencies=[Depends(parse_and_auth_request)],
)
async def get_collection(collection_name: str, dal: DataAccessLayer = Depends(get_data_access_layer)):
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
    col = await dal.get_collection_by_name(collection_name)
    if not col:
        raise HTTPException(status_code=404, detail="Collection not found")
    return collection_to_model(col)


@collections_router.patch(
    "/vectorstore/collections/{collection_name}",
    summary="Update collection info",
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
    update_fields = {}
    if body.description is not None:
        update_fields["description"] = body.description

    col = await dal.update_collection(collection_name, update_fields)
    if not col:
        raise HTTPException(status_code=404, detail="Collection not found")

    return collection_to_model(col)


@collections_router.delete(
    "/vectorstore/collections/{collection_name}",
    status_code=204,
    summary="Delete collection",
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
    success = await dal.delete_collection(collection_name)
    if not success:
        raise HTTPException(status_code=404, detail="Collection not found")
    return None
