"""Routes for creating, reading, updating, and deleting vector collections."""

from fastapi import APIRouter, Depends, HTTPException
from starlette import status

from common.fastapi.responses import (
    AUTH_RESPONSES,
    BAD_REQUEST_RESPONSE,
    NO_CONTENT_RESPONSE,
    not_found_response,
)
from gen3_embeddings.config import DEFAULT_PAGE_SIZE, logging
from gen3_embeddings.dependencies import AuthzContext, authz
from gen3_embeddings.models.helpers import collection_to_model, normalize_collection_name
from gen3_embeddings.models.schemas import (
    CollectionModel,
    CreateCollectionBody,
    PaginatedCollectionsResponse,
    UpdateCollectionBody,
)
from gen3_embeddings.params import CollectionName, Counts, Page, PageSize
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
    response_description="A page of collections you can read.",
    responses={**AUTH_RESPONSES},
    tags=["Vectorstore Collections"],
)
async def list_collections(
    page: Page = 1,
    page_size: PageSize = DEFAULT_PAGE_SIZE,
    counts: Counts = False,
    ctx: AuthzContext = Depends(authz("read")),
):
    """
    List all existing collections.

    This path names no collection, so there is no resource to check against the policy
    engine: the caller sees exactly what their `read` grants make visible, which may be
    nothing.

    Args:
        page: Page number for pagination (1-based).
        page_size: Number of items per page.
        counts: Whether to include available embeddings count.
        ctx: Authorization context for this request.

    Returns:
        PaginatedCollectionsResponse containing all collections.
    """
    offset = (page - 1) * page_size
    limit = page_size

    logging.debug(f"Listing collections, offset={offset}, limit={limit}")
    collections = await ctx.dal.list_collections(offset=offset, limit=limit)

    # If counts=true, compute per-collection embedding counts
    counts_by_id: dict[int, int] = {}
    if counts:
        for col in collections:
            counts_by_id[col.id] = await ctx.dal.count_available_embeddings_in_collection(collection=col)

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
    response_description="The collection that was created.",
    responses={**AUTH_RESPONSES, **BAD_REQUEST_RESPONSE},
    tags=["Vectorstore Collections"],
)
async def create_collection(
    body: CreateCollectionBody,
    ctx: AuthzContext = Depends(authz("create")),
):
    """
    Create a new collection.

    The collection being created is named in the body rather than the path, so the caller
    must hold `create` on the name they asked for. That check is the DAL's, against the same
    resolved grants, and it is backed by the table's RLS WITH CHECK.

    Args:
        body: Request body containing collection_name, description, and dimensions.
        ctx: Authorization context for this request.

    Returns:
        CollectionModel for the created collection.
    """
    logging.debug(f"Creating collection: {body}...")

    try:
        normalized_name = normalize_collection_name(body.collection_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    col = await ctx.dal.create_collection(
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
    response_description="The requested collection's metadata.",
    responses={
        **AUTH_RESPONSES,
        **not_found_response("Collection"),
    },
    tags=["Vectorstore Collections"],
)
async def get_collection(
    collection_name: CollectionName,
    counts: Counts = False,
    ctx: AuthzContext = Depends(authz("read")),
):
    """
    Read information about a specific collection.

    Args:
        collection_name: Name of the collection.
        counts: Whether to include available embeddings count.
        ctx: Authorization context for this request.

    Returns:
        CollectionModel representing the collection.

    Raises:
        HTTPException: 404 if collection is not found.
    """
    logging.debug(f"Getting collection: {collection_name}...")

    col = await ctx.dal.get_collection_by_name(collection_name)

    if not col:
        raise HTTPException(status_code=404, detail="Collection not found")
    available_embeddings_count = None

    if counts:
        available_embeddings_count = await ctx.dal.count_available_embeddings_in_collection(col)
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
    response_model=CollectionModel,
    response_description="The collection as it stands after the update.",
    responses={
        **AUTH_RESPONSES,
        **BAD_REQUEST_RESPONSE,
        **not_found_response("Collection"),
    },
    tags=["Vectorstore Collections"],
    response_model_exclude_none=True,
)
async def update_collection(
    collection_name: CollectionName,
    body: UpdateCollectionBody,
    ctx: AuthzContext = Depends(authz("update")),
):
    """
    Update mutable metadata fields for a collection.

    Args:
        collection_name: Name of the collection to update.
        body: Request body containing fields to update (e.g., description).
        ctx: Authorization context for this request.

    Returns:
        CollectionModel for the updated collection.

    Raises:
        HTTPException: 404 if collection is not found.
    """
    logging.debug(f"Updating collection: {collection_name} with description={body.description}...")
    col = await ctx.dal.update_collection(collection_name=collection_name, description=body.description)
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
)
async def delete_collection(
    collection_name: CollectionName,
    ctx: AuthzContext = Depends(authz("delete")),
):
    """
    Delete a collection by name.

    Args:
        collection_name: Name of the collection to delete.
        ctx: Authorization context for this request.

    Returns:
        None on success.

    Raises:
        HTTPException: 404 if collection is not found.
    """
    success = await ctx.dal.delete_collection(collection_name)
    logging.info(f"Deleted collection: {collection_name}.")
    if not success:
        raise HTTPException(status_code=404, detail="Collection not found")
    return None
