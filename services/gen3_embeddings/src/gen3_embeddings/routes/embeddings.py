"""Routes for creating, reading, updating, and deleting embeddings within collections."""

from typing import cast
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from starlette import status

from common.fastapi.responses import (
    AUTH_RESPONSES,
    BAD_REQUEST_RESPONSE,
    NO_CONTENT_RESPONSE,
    not_found_response,
)
from gen3_embeddings.auth import get_authz_resource_path_from_collection_name
from gen3_embeddings.config import DEFAULT_PAGE_SIZE, logging
from gen3_embeddings.dependencies import AuthzContext, authz
from gen3_embeddings.models.helpers import (
    embedding_to_result,
    normalize_authz,
)
from gen3_embeddings.models.schemas import (
    CreateEmbeddingsBody,
    EmbeddingResponse,
    PaginatedEmbeddingResponse,
    SingleEmbeddingResult,
    UpdateEmbeddingBody,
)
from gen3_embeddings.params import AiModel, CollectionName, Page, PageSize
from gen3_embeddings.routes.helpers import dual_path

embeddings_router = APIRouter()


@dual_path(
    embeddings_router,
    "get",
    "/vectorstore/collections/{collection_name}/embeddings/{embedding_uuid}",
    response_model=SingleEmbeddingResult,
    response_model_exclude_none=True,
    summary="Read embedding from collection",
    description="Returns a single embedding from a collection, looked up by its UUID.",
    responses={
        **AUTH_RESPONSES,
        **not_found_response("Collection or embedding"),
    },
    tags=["Embeddings"],
)
async def get_embedding_from_collection(
    collection_name: CollectionName,
    embedding_uuid: UUID,
    ctx: AuthzContext = Depends(authz("read")),
):
    """
    Read a single embedding from a specific collection.

    Args:
        collection_name: Name of the collection.
        embedding_uuid: UUID of the embedding.
        ctx: Authorization context for this request.

    Returns:
        SingleEmbeddingResult

    Raises:
        HTTPException: 404 if the collection or embedding is not found.
    """
    collection = await ctx.dal.get_collection_by_name(collection_name)
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")

    emb = await ctx.dal.get_embedding_by_collection_and_id(
        collection=collection,
        embedding_id=embedding_uuid,
    )
    if not emb:
        raise HTTPException(status_code=404, detail="Embedding not found")

    return embedding_to_result(emb=emb, collection=collection, exclude_info=False)


@dual_path(
    embeddings_router,
    "put",
    "/vectorstore/collections/{collection_name}/embeddings/{embedding_uuid}",
    response_model=SingleEmbeddingResult,
    response_model_exclude_none=True,
    summary="Update embedding in collection",
    description=(
        "Replaces the vector and/or metadata of an existing embedding, looked up by its UUID. "
        "A new vector must match the collection's dimensions."
    ),
    responses={
        **AUTH_RESPONSES,
        **BAD_REQUEST_RESPONSE,
        **not_found_response("Collection or embedding"),
    },
    tags=["Embeddings"],
)
async def update_embedding_in_collection(
    collection_name: CollectionName,
    embedding_uuid: UUID,
    body: UpdateEmbeddingBody,
    ctx: AuthzContext = Depends(authz("update")),
):
    """
    Update the embedding vector for a given collection and embedding ID.

    Args:
        collection_name: Name of the collection.
        embedding_uuid: UUID of the embedding.
        body: Request body containing the new embedding vector and/or metadata.
        ctx: Authorization context for this request.

    Returns:
        SingleEmbeddingResult containing the updated embedding.

    Raises:
        HTTPException: 404 if the collection is not found; 400 if update fails.
    """
    collection = await ctx.dal.get_collection_by_name(collection_name)
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")

    # If embedding is provided, enforce dimensions
    if body.embedding is not None and len(body.embedding) != collection.dimensions:
        raise HTTPException(status_code=400, detail="Embedding dimension mismatch")

    # If authz is being updated, check update permission on the new authz path
    authz_path = normalize_authz(body.authz)
    if authz_path is not None:
        if authz_path == "":
            raise HTTPException(status_code=400, detail="authz cannot be empty when provided")

        await ctx.require(authz_path)

    emb = await ctx.dal.update_embedding(
        collection=collection,
        embedding_id=embedding_uuid,
        embedding=body.embedding,
        metadata=body.metadata,
        new_authz=authz_path,
    )
    if not emb:
        raise HTTPException(status_code=400, detail="Failed to update embedding")

    return embedding_to_result(emb=emb, collection=collection, exclude_info=False)


@dual_path(
    embeddings_router,
    "put",
    "/vectorstore/collections/{collection_name}/embeddings",
    response_model=EmbeddingResponse,
    summary="Create or update embeddings in collection",
    description=(
        "Creates one or more embeddings in a collection, replacing any that already exist with the "
        "same UUID. Every vector must match the collection's dimensions.\n\n"
        "Authorization is resolved per request:\n\n"
        "- If `authz` is set on the request body, that path is applied to all embeddings in the "
        "request.\n"
        "- Otherwise the collection's own path, `/vectorstore/collections/{collection_name}`, is "
        "used.\n"
        "- You need both `create` and `update` permission on the collection path and on the "
        "resulting embedding paths."
    ),
    responses={
        **AUTH_RESPONSES,
        **BAD_REQUEST_RESPONSE,
        **not_found_response("Collection"),
    },
    tags=["Embeddings"],
)
async def put_embeddings_in_collection(
    collection_name: CollectionName,
    body: CreateEmbeddingsBody,
    ai_model: AiModel = None,
    exclude_info: bool = Query(False, alias="exclude_info"),
    # An upsert both creates and updates, so both are demanded on the collection. `update`
    # is the primary action, so it is the one that scopes row-level security.
    ctx: AuthzContext = Depends(authz("update", also_require=("create",))),
):
    """
    Create or update one or more embeddings in a specific collection.

    This minimal implementation only accepts raw numeric vectors.

    - If `body.authz` is provided, use the authz path for all embeddings.
    - If not provided, default authz is `/vectorstore/collections/{collection_name}`.
    - Check `create` and `update` permissions on:
        - the collection authz path, and
        - the embedding authz paths.

    Args:
        collection_name: Name of the collection.
        body: Request body containing a list of embedding vectors.
        ai_model: Optional model name; not used in this minimal version.
        exclude_info: If True, omit the 'info' block in each embedding result.
        ctx: Authorization context for this request.

    Returns:
        EmbeddingResponse containing the created embeddings.

    Raises:
        HTTPException: 404 if collection is not found; 400 if dimensions mismatch.

    """
    collection = await ctx.dal.get_collection_by_name(collection_name)
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")

    if not body.embeddings:
        raise HTTPException(status_code=400, detail="embeddings must be a non-empty array")

    default_collection_authz = get_authz_resource_path_from_collection_name(collection_name)
    embedding_authz_path = normalize_authz(body.authz) or default_collection_authz

    # The collection path was already checked for both actions by the dependency. A
    # body-supplied path is a different resource, so it needs the same pair of checks.
    if embedding_authz_path != default_collection_authz:
        logging.debug(f"authorizing `create` and `update` on embedding authz path {embedding_authz_path}")
        await ctx.require(embedding_authz_path, action="create")
        await ctx.require(embedding_authz_path, action="update")

    vectors_no_id: list[list[float]] = []
    metadata_list_no_id: list[dict] = []
    items_with_id: list[tuple[UUID, list[float], dict]] = []

    for item in body.embeddings:
        emb = item.embedding
        meta = item.metadata or {}

        # `EmbeddingToCreate.embedding` is `Vector | TextChunks`, so a list of text chunks
        # validates too. Embedding raw text is not wired up yet, so reject it here rather
        # than letting str elements reach the vector column. The union is
        # `list[float] | list[str]`, so the list is homogeneous and element 0 decides.
        if emb and isinstance(emb[0], str):
            raise HTTPException(status_code=400, detail="Raw text embedding not implemented")
        vector = cast(list[float], emb)

        if len(vector) != collection.dimensions:
            raise HTTPException(
                status_code=400,
                detail=f"Embedding dimension mismatch. Given {len(vector)}, expected {collection.dimensions} for collection",
            )

        if item.embedding_id is not None:
            items_with_id.append((item.embedding_id, vector, meta))
        else:
            vectors_no_id.append(vector)
            metadata_list_no_id.append(meta)

    logging.debug(f"PUT (upsert) embeddings in collection.id: `{collection.id}`...")

    updated_from_ids = []
    for i, (emb_id, emb_vec, meta) in enumerate(items_with_id):
        emb = await ctx.dal.update_embedding(
            collection=collection,
            embedding_id=emb_id,
            embedding=emb_vec,
            metadata=meta,
            new_authz=embedding_authz_path,
        )
        if not emb:
            # If embedding_id not found or RLS denied
            raise HTTPException(
                status_code=400,
                detail=f"Failed to update embedding with id {emb_id}",
            )
        updated_from_ids.append(emb)

    created_or_updated = []
    if vectors_no_id:
        created_or_updated = await ctx.dal.upsert_embeddings_bulk(
            collection=collection,
            embeddings=vectors_no_id,
            authz=embedding_authz_path,
            metadata_list=metadata_list_no_id,
        )

    results: list[SingleEmbeddingResult] = []

    # Iterate again over body.embeddings and map each to either updated_from_ids
    # or created_or_updated in the same order.
    id_idx = 0
    noid_idx = 0

    for i, item in enumerate(body.embeddings):
        if item.embedding_id is not None:
            emb = updated_from_ids[id_idx]
            id_idx += 1
        else:
            emb = created_or_updated[noid_idx]
            noid_idx += 1

        results.append(
            embedding_to_result(
                emb=emb,
                collection=collection,
                input_index=i,
                exclude_info=exclude_info,
            )
        )

    return EmbeddingResponse(embeddings=results)


@dual_path(
    embeddings_router,
    "delete",
    "/vectorstore/collections/{collection_name}/embeddings/{embedding_uuid}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete embedding from collection",
    description=(
        "Permanently deletes a single embedding from a collection, looked up by its UUID. This cannot be undone."
    ),
    responses={
        **NO_CONTENT_RESPONSE,
        **AUTH_RESPONSES,
        **not_found_response("Collection or embedding"),
    },
    tags=["Embeddings"],
)
async def delete_embedding(
    collection_name: CollectionName,
    embedding_uuid: UUID,
    ctx: AuthzContext = Depends(authz("delete")),
):
    """
    Delete an embedding from a specific collection.

    Args:
        collection_name: Name of the collection.
        embedding_uuid: UUID of the embedding to delete.
        ctx: Authorization context for this request.

    Returns:
        None on success.

    Raises:
        HTTPException: 404 if the collection or embedding is not found.
    """
    collection = await ctx.dal.get_collection_by_name(collection_name)
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")

    success = await ctx.dal.delete_embedding(
        collection=collection,
        embedding_id=embedding_uuid,
    )
    if not success:
        raise HTTPException(status_code=404, detail="Embedding not found or already deleted")

    return None


@dual_path(
    embeddings_router,
    "get",
    "/vectorstore/collections/{collection_name}/embeddings",
    response_model=PaginatedEmbeddingResponse,
    summary="Read all embeddings from collection",
    description=(
        "Returns the embeddings in a collection, one page at a time. "
        "Set `exclude_info=true` to omit the `info` block from each result."
    ),
    responses={
        **AUTH_RESPONSES,
        **not_found_response("Collection"),
    },
    tags=["Embeddings"],
)
async def list_embeddings_in_collection(
    collection_name: CollectionName,
    exclude_info: bool = Query(False, alias="exclude_info"),
    page: Page = 1,
    page_size: PageSize = DEFAULT_PAGE_SIZE,
    ctx: AuthzContext = Depends(authz("read")),
):
    """
    List all embeddings within a specific collection.

    Args:
        collection_name: Name of the collection.
        exclude_info: If True, omit the 'info' block in each embedding result.
        page: Page number for pagination (1-based).
        page_size: Number of items per page.
        ctx: Authorization context for this request.

    Returns:
        PaginatedEmbeddingResponse containing all embeddings in the collection.

    Raises:
        HTTPException: 404 if the collection is not found.
    """
    collection = await ctx.dal.get_collection_by_name(collection_name)
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")

    offset = (page - 1) * page_size
    limit = page_size

    embs = await ctx.dal.list_embeddings_in_collection(
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


@dual_path(
    embeddings_router,
    "post",
    "/vectorstore/collections/{collection_name}/embeddings",
    response_model=EmbeddingResponse,
    summary="Create embeddings in collection",
    description=(
        "Creates one or more embeddings in a collection. Every vector must match the collection's "
        "dimensions. Use `PUT` on this path instead if you want existing embeddings to be "
        "replaced rather than rejected.\n\n"
        "Authorization is resolved per request:\n\n"
        "- If `authz` is set on the request body, those paths are applied to all embeddings in the "
        "request.\n"
        "- Otherwise the collection's own path, `/vectorstore/collections/{collection_name}`, is "
        "used.\n"
        "- You need `create` permission on the collection path and on the resulting embedding "
        "paths."
    ),
    responses={
        **AUTH_RESPONSES,
        **BAD_REQUEST_RESPONSE,
        **not_found_response("Collection"),
    },
    tags=["Embeddings"],
)
async def create_embeddings_in_collection(
    collection_name: CollectionName,
    body: CreateEmbeddingsBody,
    ai_model: AiModel = None,
    exclude_info: bool = Query(False, alias="exclude_info"),
    ctx: AuthzContext = Depends(authz("create")),
):
    """
    Create one or more embeddings in a specific collection.

    This minimal implementation only accepts raw numeric vectors.

    - If `body.authz` is provided (list of authz paths), use those authz paths for all embeddings.
    - If not provided, default authz is `/vectorstore/collections/{collection_name}`.
    - Check `create` on:
        - the collection authz path, and
        - the embedding authz paths.

    Args:
        collection_name: Name of the collection.
        body: Request body containing a list of embedding vectors.
        ai_model: Optional model name; not used in this minimal version.
        exclude_info: If True, omit the 'info' block in each embedding result.
        ctx: Authorization context for this request.

    Returns:
        EmbeddingResponse containing the created embeddings.

    Raises:
        HTTPException: 404 if collection is not found; 400 if dimensions mismatch.
    """
    collection = await ctx.dal.get_collection_by_name(collection_name)
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")

    if not body.embeddings:
        raise HTTPException(status_code=400, detail="embeddings must be a non-empty array")

    default_collection_authz = get_authz_resource_path_from_collection_name(collection_name)
    embedding_authz_path = normalize_authz(body.authz) or default_collection_authz

    # The collection path was already checked by the dependency; a body-supplied path is a
    # different resource and needs its own check.
    if embedding_authz_path != default_collection_authz:
        logging.debug(f"authorizing `create` on embedding authz path {embedding_authz_path}")
        await ctx.require(embedding_authz_path)

    vectors: list[list[float]] = []
    metadata_list: list[dict] = []

    for item in body.embeddings:
        emb = item.embedding
        meta = item.metadata or {}

        # See the matching guard in the PUT handler: text chunks validate against the
        # `Vector | TextChunks` union but cannot be stored as a vector yet.
        if emb and isinstance(emb[0], str):
            raise HTTPException(status_code=400, detail="Raw text embedding not implemented")
        vector = cast(list[float], emb)

        if len(vector) != collection.dimensions:
            raise HTTPException(
                status_code=400,
                detail=f"Embedding dimension mismatch. Given {len(vector)}, expected {collection.dimensions} for collection",
            )

        vectors.append(vector)
        metadata_list.append(meta)

    logging.debug(f"Creating embeddings in collection.id: `{collection.id}`...")

    created = await ctx.dal.create_embeddings_bulk(
        collection=collection,
        embeddings=vectors,
        authz=embedding_authz_path,
        metadata_list=metadata_list,
    )

    results: list[SingleEmbeddingResult] = []
    for i, emb in enumerate(created):
        res = embedding_to_result(emb=emb, collection=collection, input_index=i, exclude_info=exclude_info)
        results.append(res)

    return EmbeddingResponse(embeddings=results)
