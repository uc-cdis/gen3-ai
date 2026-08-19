"""
Routes for bulk embedding reads.

These endpoints return vectors in a binary encoding rather than as JSON float arrays,
which is cheaper for large vectors. They are declared `POST` so the UUID list can be
sent in the request body, but they only read.
"""

from uuid import UUID

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request

from common.fastapi.responses import (
    AUTH_RESPONSES,
    not_found_response,
)
from gen3_embeddings.database.db import (
    DataAccessLayer,
    get_data_access_layer_for_read_operations,
)
from gen3_embeddings.database.models import Collection
from gen3_embeddings.models.helpers import (
    collection_to_model,
    embedding_to_binary_result,
)
from gen3_embeddings.models.schemas import (
    EmbeddingResponseBinary,
    EmbeddingResponseBinaryWithCollections,
    SingleEmbeddingResultBinary,
)
from gen3_embeddings.routes.helpers import dual_path

embeddings_bulk_router = APIRouter()


@dual_path(
    embeddings_bulk_router,
    "post",
    "/embeddings/bulk",
    tags=["Embeddings (Bulk Read)"],
    summary="Read select embeddings from unknown collections",
    description=(
        "Returns the requested embeddings by UUID without needing to know which collection each "
        "one belongs to, along with the metadata for every collection involved. Vectors are "
        "returned in a binary encoding. UUIDs that do not resolve are omitted from the response "
        "rather than raising an error.\n\n"
        "This uses `POST` so the UUID list can be sent in the request body, but it does not "
        "create anything."
    ),
    responses={**AUTH_RESPONSES},
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
            precision=col.vector_type.precision,
        )
        results.append(res)

    return EmbeddingResponseBinaryWithCollections(
        embeddings=results,
        collections=[collection_to_model(col) for col in collections.values()],
    )


@dual_path(
    embeddings_bulk_router,
    "post",
    "/vectorstore/collections/{collection_name}/embeddings/bulk",
    # do NOT add a response model class in this path operation decorator
    # just rely on it in the typed return. FastAPI docs say this is more performant
    summary="Read select embeddings from collection",
    description=(
        "Returns the requested embeddings by UUID from a single known collection. Vectors are "
        "returned in a binary encoding. UUIDs that do not resolve within the collection are "
        "omitted from the response rather than raising an error.\n\n"
        "This uses `POST` so the UUID list can be sent in the request body, but it does not "
        "create anything."
    ),
    responses={
        **AUTH_RESPONSES,
        **not_found_response("Collection"),
    },
    tags=["Embeddings (Bulk Read)"],
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

    rows = await dal.get_embeddings_bulk_from_collection_ordered(
        embedding_ids=embedding_uuids,
        collection=collection,
    )

    precision = collection.vector_type.precision

    binary_results = [
        embedding_to_binary_result(
            emb=emb,
            collection=collection,
            input_index=input_index,
            exclude_info=exclude_info,
            precision=precision,
        )
        for input_index, emb in rows
    ]

    return EmbeddingResponseBinary(embeddings=binary_results, count=len(binary_results))
