from fastapi import APIRouter, Depends, HTTPException, Query, Request

from gen3_embeddings.auth import (
    get_allowed_authz_for_request,
    parse_and_auth_request,
)
from gen3_embeddings.db import Collection, DataAccessLayer, Embedding, get_data_access_layer
from gen3_embeddings.models.helpers import collection_to_model, embedding_to_result
from gen3_embeddings.models.schemas import (
    SearchRequestBody,
    SearchResponse,
    SingleEmbeddingResult,
    SingleSearchResult,
)

vectorstore_search_router = APIRouter(tags=["Vectorstore Search"])


@vectorstore_search_router.post(
    "/vectorstore/collections/{collection_name}/search",
    response_model=SearchResponse,
    summary="Search embeddings in collection",
    dependencies=[Depends(parse_and_auth_request)],
)
async def search_in_collection(
    request: Request,
    body: SearchRequestBody,
    collection_name: str,
    ai_model: str | None = Query(None, alias="ai_model"),
    no_embeddings_info: bool = Query(False, alias="no_embeddings_info"),
    dal: DataAccessLayer = Depends(get_data_access_layer),
):
    """
    TODO: support for ai_model
    TODO: raw text search

    Perform a vector search within a specific collection.

    Args:
        request: The request object.
        body: SearchRequestBody containing the query vector and parameters.
        collection_name: Name of the collection to search.
        ai_model: Optional model name; not used in this minimal implementation.
        no_embeddings_info: If True, omit the 'info' block in each embedding result.
        dal: Data access layer dependency.

    Returns:
        SearchResponse containing search hits for this collection.

    Raises:
        HTTPException: 404 if collection is not found; 400 if input is invalid.
    """
    collection = await dal.get_collection_by_name(collection_name)
    if not collection:
        raise HTTPException(status_code=404, detail="collection not found")

    if isinstance(body.input, str):
        raise HTTPException(status_code=400, detail="Raw text search not implemented")

    query_vector = body.input
    if len(query_vector) != collection.dimensions:
        raise HTTPException(status_code=400, detail="Input vector dimension mismatch")

    allowed_authz = await get_allowed_authz_for_request(request)

    rows = await dal.search_embeddings_in_collection(
        collection_id=collection.id,
        query_vector=query_vector,
        top_k=body.top_k,
        score_range=body.range,
        filters=body.filters,
        allowed_authz=allowed_authz,
    )

    results: list[SingleSearchResult] = []
    for row in rows:
        emb = Embedding.from_record(row)
        sim = row["similarity_score"]
        emb_res = embedding_to_result(emb=emb, collection=collection, include_info=(not no_embeddings_info))
        if isinstance(emb_res, SingleEmbeddingResult):
            results.append(
                SingleSearchResult(
                    id=emb.embedding_id,
                    similarity_score=sim,
                    embedding=emb_res.model_dump(exclude_none=True),
                )
            )

    return SearchResponse(
        embeddings=results,
        collections=[collection_to_model(collection)],
    )


@vectorstore_search_router.post(
    "/vectorstore/search",
    response_model=SearchResponse,
    summary="Search embeddings across unknown collections",
)
async def search_across_collections(
    request: Request,
    body: SearchRequestBody,
    collections: str | None = Query(None, alias="collections"),
    ai_model: str | None = Query(None, alias="ai_model"),
    no_embeddings_info: bool = Query(False, alias="no_embeddings_info"),
    dal: DataAccessLayer = Depends(get_data_access_layer),
):
    """
    TODO: support for ai_model
    TODO: how to handle diffs in dimensions? current logic is not sufficient.

    Perform a vector search across multiple collections.

    Args:
        request: The request object.
        body: SearchRequestBody containing the query vector and parameters.
        collections: Optional comma-separated list of collection names to restrict the search.
        ai_model: Optional model name; not used in this minimal implementation.
        no_embeddings_info: If True, omit the 'info' block in each embedding result.
        dal: Data access layer dependency.

    Returns:
        SearchResponse containing search hits across collections.

    Raises:
        HTTPException: 400 if invalid collections are specified or input is invalid.
    """
    if collections:
        names = [v.strip() for v in collections.split(",") if v.strip()]
        collections_list: list[Collection] = []
        for name in names:
            await parse_and_auth_request(request, name)
            col = await dal.get_collection_by_name(name)
            if not col:
                raise HTTPException(status_code=400, detail=f"Invalid collection: {name}")
            collections_list.append(col)
    else:
        collections_list = await dal.list_collections()

    if not collections_list:
        return SearchResponse(embeddings=[])

    if isinstance(body.input, str):
        raise HTTPException(status_code=400, detail="Raw text search not implemented")

    dims = collections_list[0].dimensions
    if len(body.input) != dims:
        raise HTTPException(status_code=400, detail="Input vector dimension mismatch")

    collection_ids = [col.id for col in collections_list]

    allowed_authz = await get_allowed_authz_for_request(request)

    rows = await dal.search_embeddings_across_collections(
        collection_ids=collection_ids,
        query_vector=body.input,
        top_k=body.top_k,
        score_range=body.range,
        filters=body.filters,
        allowed_authz=allowed_authz,
    )

    collection_by_id = {col.id: col for col in collections_list}

    results: list[SingleSearchResult] = []
    hit_collection_ids: set[int] = set()
    for row in rows:
        emb = Embedding.from_record(row)
        col = collection_by_id.get(emb.collection_id)
        if not col:
            continue
        hit_collection_ids.add(col.id)
        sim = row["similarity_score"]
        emb_res = embedding_to_result(emb, collection=col, include_info=(not no_embeddings_info))
        if isinstance(emb_res, SingleEmbeddingResult):
            results.append(
                SingleSearchResult(
                    id=emb.embedding_id,
                    similarity_score=sim,
                    embedding=emb_res.model_dump(exclude_none=True),
                )
            )

    hit_collections = [collection_to_model(collection_by_id[cid]) for cid in hit_collection_ids]

    return SearchResponse(embeddings=results, collections=hit_collections)
