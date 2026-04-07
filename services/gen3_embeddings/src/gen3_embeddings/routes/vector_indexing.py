from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from gen3_embeddings.auth import parse_and_auth_request
from gen3_embeddings.db import DataAccessLayer, get_data_access_layer
from gen3_embeddings.models.helpers import vector_index_to_model
from gen3_embeddings.models.schemas import CreateIndexBody, UpdateIndexBody, VectorIndexModel, VectorIndiceNamesModel

vector_indexing_router = APIRouter(tags=["Vector Indexing"])


@vector_indexing_router.get(
    "/vector/indices",
    response_model=VectorIndiceNamesModel,
    summary="Read all indices",
)
async def list_indices(request: Request, dal: DataAccessLayer = Depends(get_data_access_layer)):
    """
    List all existing vector indices.

    Args:
        request: The request object
        dal: Data access layer dependency.

    Returns:
        VectorIndiceNamesModel containing all indices.
    """
    await parse_and_auth_request(request, "")
    indices = await dal.list_vector_indices()
    return VectorIndiceNamesModel(vector_indices=[vector_index_to_model(idx) for idx in indices])


@vector_indexing_router.post(
    "/vector/indices",
    response_model=VectorIndexModel,
    summary="Create index",
)
async def create_index(
    request: Request,
    body: CreateIndexBody,
    dal: DataAccessLayer = Depends(get_data_access_layer),
):
    """
    Create a new vector index.

    Args:
        request: The request object
        body: Request body containing index_name, description, and dimensions.
        dal: Data access layer dependency.

    Returns:
        VectorIndexModel for the created index.
    """
    await parse_and_auth_request(request, body.index_name)
    idx = await dal.create_vector_index(
        index_name=body.index_name,
        description=body.description,
        dimensions=body.dimensions,
    )
    return vector_index_to_model(idx)


@vector_indexing_router.get(
    "/vector/indices/{index_name}",
    response_model=VectorIndexModel,
    summary="Read index info",
    dependencies=[Depends(parse_and_auth_request)],
)
async def get_index(index_name: str, dal: DataAccessLayer = Depends(get_data_access_layer)):
    """
    Read information about a specific vector index.

    Args:
        index_name: Name of the vector index.
        dal: Data access layer dependency.

    Returns:
        VectorIndexModel representing the index.

    Raises:
        HTTPException: 404 if index is not found.
    """
    idx = await dal.get_vector_index_by_name(index_name)
    if not idx:
        raise HTTPException(status_code=404, detail="Index not found")
    return vector_index_to_model(idx)


@vector_indexing_router.patch(
    "/vector/indices/{index_name}",
    summary="Update index info",
    dependencies=[Depends(parse_and_auth_request)],
)
async def update_index(
    index_name: str,
    body: UpdateIndexBody,
    dal: DataAccessLayer = Depends(get_data_access_layer),
):
    """
    Update mutable metadata fields for a vector index.

    Args:
        index_name: Name of the index to update.
        body: Request body containing fields to update (e.g., description).
        dal: Data access layer dependency.

    Returns:
        A simple success status dict.

    Raises:
        HTTPException: 404 if index is not found.
    """
    update_fields = {}
    if body.description is not None:
        update_fields["description"] = body.description

    idx = await dal.update_vector_index(index_name, update_fields)
    if not idx:
        raise HTTPException(status_code=404, detail="Index not found")

    return {"status": "success"}


@vector_indexing_router.delete(
    "/vector/indices/{index_name}",
    status_code=204,
    summary="Delete index",
    dependencies=[Depends(parse_and_auth_request)],
)
async def delete_index(index_name: str, dal: DataAccessLayer = Depends(get_data_access_layer)):
    """
    Delete a vector index by name.

    Args:
        index_name: Name of the vector index to delete.
        dal: Data access layer dependency.

    Returns:
        None on success.

    Raises:
        HTTPException: 404 if index is not found.
    """
    success = await dal.delete_vector_index(index_name)
    if not success:
        raise HTTPException(status_code=404, detail="Index not found")
    return None
