from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from gen3_embeddings.db import DataAccessLayer, VectorIndex, get_embedding_dal
from gen3_embeddings.models.helpers import (
    embedding_to_single_result,
    vector_index_to_model,
)
from gen3_embeddings.models.schemas import (
    EmbeddingResponse,
    EmbeddingResponseNoIndex,
    SingleEmbeddingResult,
    SingleEmbeddingResultNoIndex,
    UpdateEmbeddingBody,
)

embeddings_router = APIRouter(tags=["Embeddings", "Embeddings (Bulk)"])


@embeddings_router.get(
    "/embeddings/{embedding_uuid}",
    response_model=SingleEmbeddingResult,
    summary="Read embedding from unknown index",
)
async def get_embedding(
    embedding_uuid: UUID,
    dal: DataAccessLayer = Depends(get_embedding_dal),
):
    """
    Read a single embedding when the index is not known in advance.

    Looks up the embedding by ID and then resolves its index so that
    index information can be returned in the response.

    Args:
        embedding_uuid: UUID of the embedding to fetch.
        dal: Data access layer dependency.

    Returns:
        SingleEmbeddingResult including index metadata.

    Raises:
        HTTPException: 404 if the embedding or associated index is not found.
    """
    emb = await dal.get_embedding_by_id(embedding_uuid)
    if not emb:
        raise HTTPException(status_code=404, detail="Embedding not found")

    index = await dal.get_vector_index_by_id(emb.vector_index_id)
    if not index:
        raise HTTPException(status_code=404, detail="Index not found")

    return embedding_to_single_result(emb=emb, index=index, include_index=True)


@embeddings_router.get(
    "/vector/indices/{index_name}/embeddings/{embedding_uuid}",
    response_model=SingleEmbeddingResult,
    summary="Read embedding from index",
)
async def get_embedding_from_index(
    index_name: str,
    embedding_uuid: UUID,
    dal: DataAccessLayer = Depends(get_embedding_dal),
):
    """
    Read a single embedding from a specific index.

    Args:
        index_name: Name of the vector index.
        embedding_uuid: UUID of the embedding.
        dal: Data access layer dependency.

    Returns:
        SingleEmbeddingResult including index metadata.

    Raises:
        HTTPException: 404 if the index or embedding is not found.
    """
    index = await dal.get_vector_index_by_name(index_name)
    if not index:
        raise HTTPException(status_code=404, detail="Index not found")

    emb = await dal.get_embedding_by_index_and_id(index.id, embedding_uuid)
    if not emb:
        raise HTTPException(status_code=404, detail="Embedding not found")

    return embedding_to_single_result(emb=emb, index=index, include_index=True)


@embeddings_router.put(
    "/vector/indices/{index_name}/embeddings/{embedding_uuid}",
    response_model=SingleEmbeddingResult,
    summary="Update embedding in index",
)
async def update_embedding_in_index(
    index_name: str,
    embedding_uuid: UUID,
    body: UpdateEmbeddingBody,
    dal: DataAccessLayer = Depends(get_embedding_dal),
):
    """
    Update the embedding vector for a given index and embedding ID.

    Args:
        index_name: Name of the vector index.
        embedding_uuid: UUID of the embedding.
        body: Request body containing the new embedding vector.
        dal: Data access layer dependency.

    Returns:
        SingleEmbeddingResult containing the updated embedding.

    Raises:
        HTTPException: 404 if the index is not found; 400 if update fails.
    """
    index = await dal.get_vector_index_by_name(index_name)
    if not index:
        raise HTTPException(status_code=404, detail="Index not found")

    emb = await dal.update_embedding(index.id, embedding_uuid, body.embedding)
    if not emb:
        raise HTTPException(status_code=400, detail="Failed to update embedding")

    return embedding_to_single_result(emb=emb, index=index, include_index=True)


@embeddings_router.delete(
    "/vector/indices/{index_name}/embeddings/{embedding_uuid}",
    status_code=204,
    summary="Delete embedding from index",
)
async def delete_embedding(
    index_name: str,
    embedding_uuid: UUID,
    dal: DataAccessLayer = Depends(get_embedding_dal),
):
    """
    Delete an embedding from a specific index.

    Args:
        index_name: Name of the vector index.
        embedding_uuid: UUID of the embedding to delete.
        dal: Data access layer dependency.

    Returns:
        None on success.

    Raises:
        HTTPException: 404 if the index or embedding is not found.
    """
    index = await dal.get_vector_index_by_name(index_name)
    if not index:
        raise HTTPException(status_code=404, detail="Index not found")

    success = await dal.delete_embedding(index.id, embedding_uuid)
    if not success:
        raise HTTPException(status_code=404, detail="Embedding not found or already deleted")

    return None


@embeddings_router.get(
    "/vector/indices/{index_name}/embeddings",
    response_model=EmbeddingResponseNoIndex,
    summary="Read all embeddings from index",
)
async def list_embeddings_in_index(
    index_name: str,
    no_embeddings_info: bool = Query(False, alias="no_embeddings_info"),
    dal: DataAccessLayer = Depends(get_embedding_dal),
):
    """
    List all embeddings within a specific index.

    Args:
        index_name: Name of the vector index.
        no_embeddings_info: If True, omit the 'info' block in each embedding result.
        dal: Data access layer dependency.

    Returns:
        EmbeddingResponseNoIndex containing all embeddings in the index.

    Raises:
        HTTPException: 404 if the index is not found.
    """
    index = await dal.get_vector_index_by_name(index_name)
    if not index:
        raise HTTPException(status_code=404, detail="Index not found")

    embs = await dal.list_embeddings_in_index(index.id)
    results: list[SingleEmbeddingResultNoIndex] = []

    for emb in embs:
        res = embedding_to_single_result(emb=emb, index=index, include_index=False)
        if isinstance(res, SingleEmbeddingResultNoIndex):
            if no_embeddings_info:
                res.info = None
            results.append(res)

    return EmbeddingResponseNoIndex(embeddings=results)


@embeddings_router.post(
    "/vector/indices/{index_name}/embeddings",
    response_model=EmbeddingResponseNoIndex,
    summary="Create embeddings in index",
)
async def create_embeddings_in_index(
    index_name: str,
    embeddings: list[list[float]] = Body(..., examples=[[20.3, 230.1, 18.2], [35.3, 83.1, 13.9]]),
    ai_model: str | None = Query(None, alias="ai_model"),
    no_embeddings_info: bool = Query(False, alias="no_embeddings_info"),
    dal: DataAccessLayer = Depends(get_embedding_dal),
):
    """
    TODO: implementaion for StringArrayInput and ai_model
    TODO: auth related

    Create one or more embeddings in a specific index.

    This minimal implementation only accepts raw numeric vectors.

    Args:
        index_name: Name of the vector index.
        body: Request body containing a list of embedding vectors.
        ai_model: Optional model name; not used in this minimal version.
        no_embeddings_info: If True, omit the 'info' block in each embedding result.
        dal: Data access layer dependency.

    Returns:
        EmbeddingResponseNoIndex containing the created embeddings.

    Raises:
        HTTPException: 404 if index is not found; 400 if dimensions mismatch.
    """
    index = await dal.get_vector_index_by_name(index_name)
    if not index:
        raise HTTPException(status_code=404, detail="Index not found")

    for e in embeddings:
        if len(e) != index.dimensions:
            raise HTTPException(status_code=400, detail="Embedding dimension mismatch")

    created = await dal.create_embeddings_bulk(
        vector_index_id=index.id,
        embeddings=embeddings,
        authz_version=0,
        authz=["/data"],  # placeholder authz
        metadata_list=[{} for _ in embeddings],
    )

    results: list[SingleEmbeddingResultNoIndex] = []
    for i, emb in enumerate(created):
        res = embedding_to_single_result(emb=emb, index=index, input_index=i, include_index=False)
        if isinstance(res, SingleEmbeddingResultNoIndex):
            if no_embeddings_info:
                res.info = None
            results.append(res)

    return EmbeddingResponseNoIndex(embeddings=results)


@embeddings_router.post(
    "/embeddings/bulk",
    response_model=EmbeddingResponse,
    summary="Read select embeddings from unknown indices",
)
async def get_embeddings_bulk_unknown_indices(
    embedding_uuids: list[UUID] = Body(..., examples=["embedding_uuid_0", "embedding_uuid_1"]),
    no_embeddings_info: bool = Query(False, alias="no_embeddings_info"),
    dal: DataAccessLayer = Depends(get_embedding_dal),
):
    """
    Read a selection of embeddings by UUID across any index.

    Args:
        embedding_uuids: List of embedding UUIDs to fetch.
        no_embeddings_info: If True, omit the 'info' block for each embedding.
        dal: Data access layer dependency.

    Returns:
        EmbeddingResponse including index metadata for each embedding.
    """
    embs = await dal.get_embeddings_bulk(embedding_uuids)
    if not embs:
        return EmbeddingResponse(embeddings=[], vector_indices=[])

    emb_by_id = {e.embedding_id: e for e in embs}

    index_ids = [e.vector_index_id for e in embs]
    indices: dict[int, VectorIndex] = {}

    idx_list = await dal.get_vector_index_by_id_bulk(index_ids)

    for idx in idx_list:
        indices[idx.id] = idx

    results: list[SingleEmbeddingResult] = []
    # Preserve the original order and input index
    for input_idx, emb_id in enumerate(embedding_uuids):
        emb = emb_by_id.get(emb_id)
        if not emb:
            # Requested ID not found; skip or handle separately if desired
            continue
        idx = indices.get(emb.vector_index_id)
        if not idx:
            # Index missing for this embedding; skip
            continue

        res = embedding_to_single_result(
            emb=emb,
            index=idx,
            input_index=input_idx,
            include_index=True,
        )
        if isinstance(res, SingleEmbeddingResult):
            if no_embeddings_info:
                res.info = None
            results.append(res)

    return EmbeddingResponse(
        embeddings=results,
        vector_indices=[vector_index_to_model(idx) for idx in indices.values()],
    )


@embeddings_router.post(
    "/vector/indices/{index_name}/embeddings/bulk",
    response_model=EmbeddingResponseNoIndex,
    summary="Read select embeddings from index",
)
async def get_embeddings_bulk_from_index(
    index_name: str,
    embedding_uuids: list[UUID] = Body(..., examples=["embedding_uuid_0", "embedding_uuid_1"]),
    no_embeddings_info: bool = Query(False, alias="no_embeddings_info"),
    dal: DataAccessLayer = Depends(get_embedding_dal),
):
    """
    Read a selection of embeddings by UUID from a specific index.

    Args:
        index_name: Name of the vector index.
        embedding_uuids: List of embedding UUIDs to fetch.
        no_embeddings_info: If True, omit the 'info' block for each embedding.
        dal: Data access layer dependency.

    Returns:
        EmbeddingResponseNoIndex containing the embeddings found in the specified index.

    Raises:
        HTTPException: 404 if the index is not found.
    """
    index = await dal.get_vector_index_by_name(index_name)
    if not index:
        raise HTTPException(status_code=404, detail="Index not found")

    embs = await dal.get_embeddings_bulk(embedding_uuids)
    embs = [e for e in embs if e.vector_index_id == index.id]

    emb_by_id = {e.embedding_id: e for e in embs}

    results: list[SingleEmbeddingResultNoIndex] = []
    # Preserve original order and input index
    for input_idx, emb_id in enumerate(embedding_uuids):
        emb = emb_by_id.get(emb_id)
        if not emb:
            continue

        res = embedding_to_single_result(
            emb=emb,
            index=index,
            input_index=input_idx,
            include_index=False,
        )
        if isinstance(res, SingleEmbeddingResultNoIndex):
            if no_embeddings_info:
                res.info = None
            results.append(res)

    return EmbeddingResponseNoIndex(embeddings=results)
