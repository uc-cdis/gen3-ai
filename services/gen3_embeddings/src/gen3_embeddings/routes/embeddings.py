from uuid import UUID

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request

from gen3_embeddings.auth import (
    authorize_request,
    get_authz_resource_path_from_collection_name,
    parse_and_auth_request,
)
from gen3_embeddings.config import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE, logging
from gen3_embeddings.database.db import (
    Collection,
    DataAccessLayer,
    get_data_access_layer,
    get_data_access_layer_for_read_operations,
)
from gen3_embeddings.models.helpers import (
    collection_to_model,
    embedding_to_binary_result,
    embedding_to_result,
)
from gen3_embeddings.models.schemas import (
    CreateEmbeddingsBody,
    EmbeddingResponse,
    EmbeddingResponseBinary,
    EmbeddingResponseBinaryWithCollections,
    PaginatedEmbeddingResponse,
    SingleEmbeddingResult,
    SingleEmbeddingResultBinary,
    UpdateEmbeddingBody,
)

embeddings_router = APIRouter()


@embeddings_router.get(
    "/vectorstore/collections/{collection_name}/embeddings/{embedding_uuid}",
    response_model=SingleEmbeddingResult,
    response_model_exclude_none=True,
    summary="Read embedding from collection",
    tags=["Embeddings"],
    dependencies=[Depends(parse_and_auth_request)],
)
@embeddings_router.get(
    "/vectorstore/collections/{collection_name}/embeddings/{embedding_uuid}/",
    include_in_schema=False,
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
        request: The request object.
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

    emb = await dal.get_embedding_by_collection_and_id(
        collection=collection,
        embedding_id=embedding_uuid,
    )
    if not emb:
        raise HTTPException(status_code=404, detail="Embedding not found")

    return embedding_to_result(emb=emb, collection=collection, exclude_info=False)


@embeddings_router.put(
    "/vectorstore/collections/{collection_name}/embeddings/{embedding_uuid}",
    response_model=SingleEmbeddingResult,
    response_model_exclude_none=True,
    summary="Update embedding in collection",
    tags=["Embeddings"],
    dependencies=[Depends(parse_and_auth_request)],
)
@embeddings_router.put(
    "/vectorstore/collections/{collection_name}/embeddings/{embedding_uuid}/",
    include_in_schema=False,
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
        request: The request object.
        collection_name: Name of the collection.
        embedding_uuid: UUID of the embedding.
        body: Request body containing the new embedding vector and/or metadata.
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

    # If authz is being updated, check update permission on new authz paths
    if body.authz is not None:
        if body.authz == []:
            raise HTTPException(status_code=400, detail="authz cannot be empty when provided")

        await authorize_request(
            request=request,
            authz_access_method="update",
            authz_resources=body.authz,
        )

    emb = await dal.update_embedding(
        collection=collection,
        embedding_id=embedding_uuid,
        embedding=body.embedding,
        metadata=body.metadata,
        new_authz=body.authz,
    )
    if not emb:
        raise HTTPException(status_code=400, detail="Failed to update embedding")

    return embedding_to_result(emb=emb, collection=collection, exclude_info=False)


@embeddings_router.put(
    "/vectorstore/collections/{collection_name}/embeddings",
    response_model=EmbeddingResponse,
    summary="Create or update embeddings in collection",
    tags=["Embeddings"],
    dependencies=[Depends(parse_and_auth_request)],
)
@embeddings_router.put(
    "/vectorstore/collections/{collection_name}/embeddings/",
    include_in_schema=False,
    dependencies=[Depends(parse_and_auth_request)],
)
async def put_embeddings_in_collection(
    request: Request,
    collection_name: str,
    body: CreateEmbeddingsBody,
    ai_model: str | None = Query(None, alias="ai_model"),
    exclude_info: bool = Query(False, alias="exclude_info"),
    dal: DataAccessLayer = Depends(get_data_access_layer),
):
    """
    TODO: implementaion for StringArrayInput and ai_model

    Create or update one or more embeddings in a specific collection. If an embedding already exists (same vector), update its metadata/authz. If it does not exist, create it. Entire request is all-or-nothing.

    This minimal implementation only accepts raw numeric vectors.

    - If `body.authz` is provided (list of authz paths), use those authz paths for all embeddings.
    - If not provided, default authz is `/vectorstore/collections/{collection_name}`.
    - Check `create` and `update` permissions on:
        - the collection authz path, and
        - the embedding authz paths.

    Args:
        request: The request object.
        collection_name: Name of the collection.
        body: Request body containing a list of embedding vectors.
        ai_model: Optional model name; not used in this minimal version.
        exclude_info: If True, omit the 'info' block in each embedding result.
        dal: Data access layer dependency.

    Returns:
        EmbeddingResponse containing the created embeddings.

    Raises:
        HTTPException: 404 if collection is not found; 400 if dimensions mismatch.

    """

    collection = await dal.get_collection_by_name(collection_name)
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")

    if not body.embeddings:
        raise HTTPException(status_code=400, detail="embeddings must be a non-empty array")

    default_collection_authz = get_authz_resource_path_from_collection_name(collection_name)
    embedding_authz_paths = body.authz or [default_collection_authz]

    # 1) Check 'create' and 'update' on collection
    logging.debug(f"authorize_request for `create` on collection {default_collection_authz}")
    await authorize_request(
        request=request,
        authz_access_method="create",
        authz_resources=[default_collection_authz],
    )
    await authorize_request(
        request=request,
        authz_access_method="update",
        authz_resources=[default_collection_authz],
    )

    # 2) Check 'create' and 'update' on embedding authz paths
    if embedding_authz_paths != [default_collection_authz]:
        logging.debug(f"authorize_request for `create` and `update` on embedding authz paths {embedding_authz_paths}")
        await authorize_request(
            request=request,
            authz_access_method="create",
            authz_resources=embedding_authz_paths,
        )
        await authorize_request(
            request=request,
            authz_access_method="update",
            authz_resources=embedding_authz_paths,
        )

    vectors: list[list[float]] = []
    metadata_list: list[dict] = []

    for item in body.embeddings:
        emb = item.embedding
        meta = item.metadata or {}

        if len(emb) != collection.dimensions:
            raise HTTPException(
                status_code=400,
                detail=f"Embedding dimension mismatch. Given {len(emb)}, expected {collection.dimensions} for collection",
            )

        vectors.append([float(x) for x in emb])
        metadata_list.append(meta)

    logging.debug(f"PUT (upsert) embeddings in collection.id: `{collection.id}`...")

    created_or_updated = await dal.upsert_embeddings_bulk(
        collection=collection,
        embeddings=vectors,
        authz=embedding_authz_paths,
        metadata_list=metadata_list,
    )

    results: list[SingleEmbeddingResult] = []
    for i, emb in enumerate(created_or_updated):
        results.append(
            embedding_to_result(
                emb=emb,
                collection=collection,
                input_index=i,
                exclude_info=exclude_info,
            )
        )

    return EmbeddingResponse(embeddings=results)


@embeddings_router.delete(
    "/vectorstore/collections/{collection_name}/embeddings/{embedding_uuid}",
    status_code=204,
    summary="Delete embedding from collection",
    tags=["Embeddings"],
    dependencies=[Depends(parse_and_auth_request)],
)
@embeddings_router.delete(
    "/vectorstore/collections/{collection_name}/embeddings/{embedding_uuid}/",
    include_in_schema=False,
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
        request: The request object.
        collection_name: Name of the collection.
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

    success = await dal.delete_embedding(
        collection=collection,
        embedding_id=embedding_uuid,
    )
    if not success:
        raise HTTPException(status_code=404, detail="Embedding not found or already deleted")

    return None


@embeddings_router.get(
    "/vectorstore/collections/{collection_name}/embeddings",
    response_model=PaginatedEmbeddingResponse,
    summary="Read all embeddings from collection",
    tags=["Embeddings"],
    dependencies=[Depends(parse_and_auth_request)],
)
@embeddings_router.get(
    "/vectorstore/collections/{collection_name}/embeddings/",
    include_in_schema=False,
    dependencies=[Depends(parse_and_auth_request)],
)
async def list_embeddings_in_collection(
    request: Request,
    collection_name: str,
    exclude_info: bool = Query(False, alias="exclude_info"),
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=DEFAULT_PAGE_SIZE, le=MAX_PAGE_SIZE),
    dal: DataAccessLayer = Depends(get_data_access_layer),
):
    """
    List all embeddings within a specific collection.

    Args:
        request: The request object.
        collection_name: Name of the collection.
        exclude_info: If True, omit the 'info' block in each embedding result.
        page: Page number for pagination (1-based).
        page_size: Number of items per page.
        dal: Data access layer dependency.

    Returns:
        PaginatedEmbeddingResponse containing all embeddings in the collection.

    Raises:
        HTTPException: 404 if the collection is not found.
    """
    collection = await dal.get_collection_by_name(collection_name)
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")

    offset = (page - 1) * page_size
    limit = page_size

    embs = await dal.list_embeddings_in_collection(
        collection=collection,
        offset=offset,
        limit=limit,
    )
    results: list[SingleEmbeddingResult] = []

    for emb in embs:
        res = embedding_to_result(emb=emb, collection=collection, exclude_info=exclude_info)
        if isinstance(res, SingleEmbeddingResult):
            results.append(res)

    next_page = page + 1 if len(results) == page_size else None
    prev_page = page - 1 if page > 1 else None

    return PaginatedEmbeddingResponse(
        embeddings=results,
        page=page,
        page_size=page_size,
        next_page=next_page,
        prev_page=prev_page,
    )


@embeddings_router.post(
    "/vectorstore/collections/{collection_name}/embeddings",
    response_model=EmbeddingResponse,
    summary="Create embeddings in collection",
    tags=["Embeddings"],
    dependencies=[Depends(parse_and_auth_request)],
)
@embeddings_router.post(
    "/vectorstore/collections/{collection_name}/embeddings/",
    include_in_schema=False,
    dependencies=[Depends(parse_and_auth_request)],
)
async def create_embeddings_in_collection(
    request: Request,
    collection_name: str,
    body: CreateEmbeddingsBody,
    ai_model: str | None = Query(None, alias="ai_model"),
    exclude_info: bool = Query(False, alias="exclude_info"),
    dal: DataAccessLayer = Depends(get_data_access_layer),
):
    """

    TODO: implementaion for StringArrayInput and ai_model

    Create one or more embeddings in a specific collection.

    This minimal implementation only accepts raw numeric vectors.

    - If `body.authz` is provided (list of authz paths), use those authz paths for all embeddings.
    - If not provided, default authz is `/vectorstore/collections/{collection_name}`.
    - Check `create` on:
        - the collection authz path, and
        - the embedding authz paths.

    Args:
        request: The request object.
        collection_name: Name of the collection.
        body: Request body containing a list of embedding vectors.
        ai_model: Optional model name; not used in this minimal version.
        exclude_info: If True, omit the 'info' block in each embedding result.
        dal: Data access layer dependency.

    Returns:
        EmbeddingResponse containing the created embeddings.

    Raises:
        HTTPException: 404 if collection is not found; 400 if dimensions mismatch.
    """
    collection = await dal.get_collection_by_name(collection_name)
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")

    if not body.embeddings:
        raise HTTPException(status_code=400, detail="embeddings must be a non-empty array")

    default_collection_authz = get_authz_resource_path_from_collection_name(collection_name)
    embedding_authz_paths = body.authz or [default_collection_authz]

    # 1) Check create on collection authz
    logging.debug(f"authorize_request for `create` on collection {default_collection_authz}")
    await authorize_request(
        request=request,
        authz_access_method="create",
        authz_resources=[default_collection_authz],
    )

    # 2) Check create on embedding authz paths
    logging.debug(f"authorize_request for `create` on collection {default_collection_authz}")
    if embedding_authz_paths != [default_collection_authz]:
        await authorize_request(
            request=request,
            authz_access_method="create",
            authz_resources=embedding_authz_paths,
        )

    vectors: list[list[float]] = []
    metadata_list: list[dict] = []

    for item in body.embeddings:
        emb = item.embedding
        meta = item.metadata or {}

        if len(emb) != collection.dimensions:
            raise HTTPException(
                status_code=400,
                detail=f"Embedding dimension mismatch. Given {len(emb)}, expected {collection.dimensions} for collection",
            )

        # TODO: use numpy float16 instead
        vectors.append([float(x) for x in emb])
        metadata_list.append(meta)

    logging.debug(f"Creating embeddings in collection.id: `{collection.id}`...")

    created = await dal.create_embeddings_bulk(
        collection=collection,
        embeddings=vectors,
        authz=embedding_authz_paths,
        metadata_list=metadata_list,
    )

    results: list[SingleEmbeddingResult] = []
    for i, emb in enumerate(created):
        res = embedding_to_result(emb=emb, collection=collection, input_index=i, exclude_info=exclude_info)
        results.append(res)

    return EmbeddingResponse(embeddings=results)


# TODO: let's move /bulk endpoints to a new embeddings_bulk.py so we match the API organization better
@embeddings_router.post(
    "/embeddings/bulk",
    tags=["Embeddings (Bulk Read)"],
    summary="Read select embeddings from unknown collections",
)
@embeddings_router.post(
    "/embeddings/bulk/",
    include_in_schema=False,
)
async def get_embeddings_bulk_unknown_collections(
    request: Request,
    embedding_uuids: list[UUID] = Body(..., examples=["embedding_uuid_0", "embedding_uuid_1"]),
    exclude_info: bool = Query(False, alias="exclude_info"),
    dal: DataAccessLayer = Depends(get_data_access_layer_for_read_operations),
) -> EmbeddingResponseBinaryWithCollections:
    """
    Read a selection of embeddings by UUID across any collection.

    Args:
        request (Request): The request object.
        embedding_uuids (list[UUID]): List of embedding UUIDs to fetch.
        exclude_info (bool): If True, exclude the 'info' block for each embedding.
        dal (DataAccessLayer): Data access layer dependency.

    Returns:
        EmbeddingResponseBinaryWithCollections including collection metadata
        for each embedding.
    """

    embs = await dal.get_embeddings_bulk(
        embedding_ids=embedding_uuids,
        vector_type=None,
    )
    if not embs:
        return EmbeddingResponseBinaryWithCollections(embeddings=[], collections=[])

    emb_by_id = {e.embedding_id: e for e in embs}

    collection_ids = list({e.collection_id for e in embs})
    collections: dict[int, Collection] = {}

    col_list = await dal.get_collection_by_id_bulk(collection_ids)

    for col in col_list:
        collections[col.id] = col

    results: list[SingleEmbeddingResultBinary] = []
    # Preserve the original order and input collection
    for input_index, emb_id in enumerate(embedding_uuids):
        emb = emb_by_id.get(emb_id)
        if not emb:
            continue
        col = collections.get(emb.collection_id)
        if not col:
            continue

        res = embedding_to_binary_result(
            emb=emb,
            collection=col,
            input_index=input_index,
            exclude_info=exclude_info,
            precision="float16" if col.vector_type == "halfvec" else "float32",
        )
        results.append(res)

    return EmbeddingResponseBinaryWithCollections(
        embeddings=results,
        collections=[collection_to_model(col) for col in collections.values()],
    )


@embeddings_router.post(
    "/vectorstore/collections/{collection_name}/embeddings/bulk",
    # do NOT add a response model class in this path operation decorator
    # just rely on it in the typed return. FastAPI docs say this is more performant
    summary="Read select embeddings from collection",
    tags=["Embeddings (Bulk Read)"],
)
@embeddings_router.post(
    "/vectorstore/collections/{collection_name}/embeddings/bulk/",
    include_in_schema=False,
)
async def get_embeddings_bulk_from_collection(
    request: Request,
    collection_name: str,
    embedding_uuids: list[UUID] = Body(..., examples=["embedding_uuid_0", "embedding_uuid_1"]),
    exclude_info: bool = Query(False, alias="exclude_info"),
    dal: DataAccessLayer = Depends(get_data_access_layer_for_read_operations),
) -> EmbeddingResponseBinary:
    """
    TODO: post here but actually reading, how to hanle authz here?

    Read a selection of embeddings by UUID from a specific collection.

    Args:
        request (Request): The request object.
        collection_name (str): Name of the collection to read from.
        embedding_uuids (list[UUID]): List of embedding UUIDs to fetch.
        exclude_info (bool): If True, exclude the 'info' block for each embedding.
        dal (DataAccessLayer): Data access layer dependency.

    Returns:
        EmbeddingResponse containing the embeddings found in the specified collection.

    Raises:
        HTTPException: 404 if the collection is not found.
    """
    collection = await dal.get_collection_by_name(collection_name)

    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")

    embs = await dal.get_embeddings_bulk(
        embedding_ids=embedding_uuids,
        collection_id=collection.id,
        vector_type=collection.vector_type,
    )

    if embs:
        logging.debug(
            f"Type of emb.embedding with vector_type=`{collection.vector_type}` is `{type(embs[-1].embedding)}`"
        )

    emb_by_id = {e.embedding_id: e for e in embs}

    binary_results: list[SingleEmbeddingResultBinary] = []

    # Preserve original order and input collection
    for input_index, emb_id in enumerate(embedding_uuids):
        emb = emb_by_id.get(emb_id)
        if not emb:
            continue

        res = embedding_to_binary_result(
            emb=emb,
            collection=collection,
            input_index=input_index,
            exclude_info=exclude_info,
            precision="float16" if collection.vector_type == "halfvec" else "float32",
        )
        binary_results.append(res)

    return EmbeddingResponseBinary(embeddings=binary_results, count=len(binary_results))
