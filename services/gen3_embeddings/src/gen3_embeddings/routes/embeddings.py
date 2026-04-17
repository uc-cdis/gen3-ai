from uuid import UUID

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request

from gen3_embeddings.auth import get_authz_resource_path_from_collection_name, parse_and_auth_request
from gen3_embeddings.db import Collection, DataAccessLayer, get_data_access_layer
from gen3_embeddings.models.helpers import (
    collection_to_model,
    embedding_to_result,
)
from gen3_embeddings.models.schemas import (
    CreateEmbeddingsBody,
    EmbeddingResponse,
    SingleEmbeddingResult,
    UpdateEmbeddingBody,
)

embeddings_router = APIRouter(tags=["Embeddings", "Embeddings (Bulk)"])


@embeddings_router.get(
    "/vectorstore/collections/{collection_name}/embeddings/{embedding_uuid}",
    response_model=SingleEmbeddingResult,
    summary="Read embedding from collection",
    dependencies=[Depends(parse_and_auth_request)],
)
async def get_embedding_from_collection(
    request: Request,
    collection_name: str,
    embedding_uuid: UUID,
    dal: DataAccessLayer = Depends(get_data_access_layer),
):
    """
    Read a single embedding from a specific collection.

    Args:
        collection_name: Name of the collection.
        embedding_uuid: UUID of the embedding.
        dal: Data access layer dependency.

    Returns:
        SingleEmbeddingResult

    Raises:
        HTTPException: 404 if the collection or embedding is not found.
    """
    collection = await dal.get_collection_by_name(collection_name)
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")

    emb = await dal.get_embedding_by_collection_and_id(request, collection.id, embedding_uuid)
    if not emb:
        raise HTTPException(status_code=404, detail="Embedding not found")

    return embedding_to_result(emb=emb, collection=collection, include_info=True)


@embeddings_router.put(
    "/vectorstore/collections/{collection_name}/embeddings/{embedding_uuid}",
    response_model=SingleEmbeddingResult,
    summary="Update embedding in collection",
    dependencies=[Depends(parse_and_auth_request)],
)
async def update_embedding_in_collection(
    request: Request,
    collection_name: str,
    embedding_uuid: UUID,
    body: UpdateEmbeddingBody,
    dal: DataAccessLayer = Depends(get_data_access_layer),
):
    """
    Update the embedding vector for a given collection and embedding ID.

    Args:
        request: The request object
        collection_name: Name of the collections.
        embedding_uuid: UUID of the embedding.
        body: Request body containing the new embedding vector.
        dal: Data access layer dependency.

    Returns:
        SingleEmbeddingResult containing the updated embedding.

    Raises:
        HTTPException: 404 if the collection is not found; 400 if update fails.
    """
    collection = await dal.get_collection_by_name(collection_name)
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")

    # If embedding is provided, enforce dimensions
    if body.embedding is not None and len(body.embedding) != collection.dimensions:
        raise HTTPException(status_code=400, detail="Embedding dimension mismatch")

    emb = await dal.update_embedding(
        request=request,
        collection_id=collection.id,
        embedding_id=embedding_uuid,
        embedding=body.embedding,
        metadata=body.metadata,
    )
    if not emb:
        raise HTTPException(status_code=400, detail="Failed to update embedding")

    return embedding_to_result(emb=emb, collection=collection, include_info=True)


@embeddings_router.delete(
    "/vectorstore/collections/{collection_name}/embeddings/{embedding_uuid}",
    status_code=204,
    summary="Delete embedding from collection",
    dependencies=[Depends(parse_and_auth_request)],
)
async def delete_embedding(
    request: Request,
    collection_name: str,
    embedding_uuid: UUID,
    dal: DataAccessLayer = Depends(get_data_access_layer),
):
    """
    Delete an embedding from a specific collection.

    Args:
        request: The request object
        collection_name: Name of the collections.
        embedding_uuid: UUID of the embedding to delete.
        dal: Data access layer dependency.

    Returns:
        None on success.

    Raises:
        HTTPException: 404 if the collection or embedding is not found.
    """
    collection = await dal.get_collection_by_name(collection_name)
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")

    success = await dal.delete_embedding(request, collection.id, embedding_uuid)
    if not success:
        raise HTTPException(status_code=404, detail="Embedding not found or already deleted")

    return None


@embeddings_router.get(
    "/vectorstore/collections/{collection_name}/embeddings",
    response_model=EmbeddingResponse,
    summary="Read all embeddings from collection",
    dependencies=[Depends(parse_and_auth_request)],
)
async def list_embeddings_in_collection(
    request: Request,
    collection_name: str,
    no_embeddings_info: bool = Query(False, alias="no_embeddings_info"),
    dal: DataAccessLayer = Depends(get_data_access_layer),
):
    """
    List all embeddings within a specific collection.

    Args:
        request: The request object
        collection_name: Name of the collections.
        no_embeddings_info: If True, omit the 'info' block in each embedding result.
        dal: Data access layer dependency.

    Returns:
        EmbeddingResponseNoCollection containing all embeddings in the collection.

    Raises:
        HTTPException: 404 if the collection is not found.
    """
    collection = await dal.get_collection_by_name(collection_name)
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")

    embs = await dal.list_embeddings_in_collection(request, collection.id)
    results: list[SingleEmbeddingResult] = []

    for emb in embs:
        res = embedding_to_result(emb=emb, collection=collection, include_info=(not no_embeddings_info))
        if isinstance(res, SingleEmbeddingResult):
            results.append(res)

    return EmbeddingResponse(embeddings=results)


@embeddings_router.post(
    "/vectorstore/collections/{collection_name}/embeddings",
    response_model=EmbeddingResponse,
    summary="Create embeddings in collection",
    dependencies=[Depends(parse_and_auth_request)],
)
async def create_embeddings_in_collection(
    request: Request,
    collection_name: str,
    body: CreateEmbeddingsBody,
    ai_model: str | None = Query(None, alias="ai_model"),
    no_embeddings_info: bool = Query(False, alias="no_embeddings_info"),
    dal: DataAccessLayer = Depends(get_data_access_layer),
):
    """
    TODO: implementaion for StringArrayInput and ai_model
    TODO: auth related
    TODO: work for authz_version

    Create one or more embeddings in a specific collection.

    This minimal implementation only accepts raw numeric vectors.

    Args:
        request: The request object
        collection_name: Name of the collection.
        body: Request body containing a list of embedding vectors.
        ai_model: Optional model name; not used in this minimal version.
        no_embeddings_info: If True, omit the 'info' block in each embedding result.
        dal: Data access layer dependency.

    Returns:
        EmbeddingResponseNoCollection containing the created embeddings.

    Raises:
        HTTPException: 404 if collection is not found; 400 if dimensions mismatch.
    """
    collection = await dal.get_collection_by_name(collection_name)
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")

    if not body.embeddings:
        raise HTTPException(status_code=400, detail="embeddings must be a non-empty array")

    vectors: list[list[float]] = []
    metadata_list: list[dict] = []

    for item in body.embeddings:
        emb = item.embedding
        meta = item.metadata or {}

        # For now: only support already-embedded numeric vectors.
        if not isinstance(emb, list) or not all(isinstance(x, (int, float)) for x in emb):
            raise HTTPException(
                status_code=400,
                detail="Only numeric vector embeddings are supported at this time",
            )

        if len(emb) != collection.dimensions:
            raise HTTPException(status_code=400, detail="Embedding dimension mismatch")

        vectors.append([float(x) for x in emb])
        metadata_list.append(meta)

    created = await dal.create_embeddings_bulk(
        request=request,
        collection_id=collection.id,
        embeddings=vectors,
        authz_version=0,
        authz=[get_authz_resource_path_from_collection_name(collection_name)],
        metadata_list=metadata_list,
    )

    results: list[SingleEmbeddingResult] = []
    for i, emb in enumerate(created):
        res = embedding_to_result(emb=emb, collection=collection, input_index=i, include_info=(not no_embeddings_info))
        results.append(res)

    return EmbeddingResponse(embeddings=results)


@embeddings_router.post(
    "/embeddings/bulk",
    response_model=EmbeddingResponse,
    summary="Read select embeddings from unknown collections",
)
async def get_embeddings_bulk_unknown_collections(
    request: Request,
    embedding_uuids: list[UUID] = Body(..., examples=["embedding_uuid_0", "embedding_uuid_1"]),
    no_embeddings_info: bool = Query(False, alias="no_embeddings_info"),
    dal: DataAccessLayer = Depends(get_data_access_layer),
):
    """
    TODO: collections list is needed as return?
    TODO: update dal.get_collection_by_id_bulk. remove duplicates?

    Read a selection of embeddings by UUID across any collection.

    Args:
        request: The request object
        embedding_uuids: List of embedding UUIDs to fetch.
        no_embeddings_info: If True, omit the 'info' block for each embedding.
        dal: Data access layer dependency.

    Returns:
        EmbeddingResponse including collection metadata for each embedding.
    """
    embs = await dal.get_embeddings_bulk(request, embedding_uuids)
    if not embs:
        return EmbeddingResponse(embeddings=[], collections=[])

    emb_by_id = {e.embedding_id: e for e in embs}

    collection_ids = [e.collection_id for e in embs]
    collections: dict[int, Collection] = {}

    col_list = await dal.get_collection_by_id_bulk(collection_ids)

    for col in col_list:
        await parse_and_auth_request(request, col.collection_name)
        collections[col.id] = col

    results: list[SingleEmbeddingResult] = []
    # Preserve the original order and input collection
    for input_index, emb_id in enumerate(embedding_uuids):
        emb = emb_by_id.get(emb_id)
        if not emb:
            # Requested ID not found; skip or handle separately if desired
            continue
        col = collections.get(emb.collection_id)
        if not col:
            # Collection missing for this embedding; skip
            continue

        res = embedding_to_result(
            emb=emb,
            collection=col,
            input_index=input_index,
            include_info=(not no_embeddings_info),
        )
        if isinstance(res, SingleEmbeddingResult):
            results.append(res)

    return EmbeddingResponse(
        embeddings=results,
        collections=[collection_to_model(col) for col in collections.values()],
    )


@embeddings_router.post(
    "/vectorstore/collections/{collection_name}/embeddings/bulk",
    response_model=EmbeddingResponse,
    summary="Read select embeddings from collection",
    dependencies=[Depends(parse_and_auth_request)],
)
async def get_embeddings_bulk_from_collection(
    request: Request,
    collection_name: str,
    embedding_uuids: list[UUID] = Body(..., examples=["embedding_uuid_0", "embedding_uuid_1"]),
    no_embeddings_info: bool = Query(False, alias="no_embeddings_info"),
    dal: DataAccessLayer = Depends(get_data_access_layer),
):
    """
    Read a selection of embeddings by UUID from a specific collection.

    Args:
        request: The request object
        collection_name: Name of the collections.
        embedding_uuids: List of embedding UUIDs to fetch.
        no_embeddings_info: If True, omit the 'info' block for each embedding.
        dal: Data access layer dependency.

    Returns:
        EmbeddingResponse containing the embeddings found in the specified collection.

    Raises:
        HTTPException: 404 if the collection is not found.
    """
    collection = await dal.get_collection_by_name(collection_name)
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")

    embs = await dal.get_embeddings_bulk(request, embedding_uuids)
    embs = [e for e in embs if e.collection_id == collection.id]

    emb_by_id = {e.embedding_id: e for e in embs}

    results: list[SingleEmbeddingResult] = []
    # Preserve original order and input collection
    for input_index, emb_id in enumerate(embedding_uuids):
        emb = emb_by_id.get(emb_id)
        if not emb:
            continue

        res = embedding_to_result(
            emb=emb,
            collection=collection,
            input_index=input_index,
            include_info=(not no_embeddings_info),
        )
        if isinstance(res, SingleEmbeddingResult):
            results.append(res)

    return EmbeddingResponse(embeddings=results)
