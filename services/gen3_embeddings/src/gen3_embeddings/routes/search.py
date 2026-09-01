"""Routes for vector similarity search within and across collections."""

from fastapi import APIRouter, Depends, HTTPException, Query

from common.fastapi.responses import (
    AUTH_RESPONSES,
    BAD_REQUEST_RESPONSE,
    not_found_response,
)
from gen3_embeddings.config import MAX_COLLECTIONS_PER_SEARCH, MAX_COLLECTIONS_SEARCHED
from gen3_embeddings.database.models import Collection, Embedding
from gen3_embeddings.dependencies import AuthzContext, authz
from gen3_embeddings.models.helpers import collection_to_model, embedding_to_result
from gen3_embeddings.models.schemas import (
    SearchRequestBody,
    SearchResponse,
    SingleEmbeddingResult,
    SingleSearchResult,
    VectorType,
)
from gen3_embeddings.params import AiModel, CollectionName, RequestedCollectionNames
from gen3_embeddings.routes.helpers import dual_path

vectorstore_search_router = APIRouter()


@dual_path(
    vectorstore_search_router,
    "post",
    "/vectorstore/collections/{collection_name}/search",
    response_model=SearchResponse,
    summary="Search embeddings in collection",
    description=(
        "Finds the embeddings in a collection nearest to the query vector you provide, ordered by "
        "similarity. The query vector must have the same number of dimensions as the collection. "
        "Set `exclude_info=true` to omit the `info` block from each result."
    ),
    responses={
        **AUTH_RESPONSES,
        **BAD_REQUEST_RESPONSE,
        **not_found_response("Collection"),
    },
    tags=["Vectorstore Search"],
)
async def search_in_collection(
    body: SearchRequestBody,
    collection_name: CollectionName,
    ai_model: AiModel = None,
    exclude_info: bool = Query(False, alias="exclude_info"),
    # POST, but this only reads. The action is declared, so the verb is irrelevant.
    ctx: AuthzContext = Depends(authz("read")),
):
    """
    Perform a vector search within a specific collection.

    Args:
        body: SearchRequestBody containing the query vector and parameters.
        collection_name: Name of the collection to search.
        ai_model: Optional model name; not used in this minimal implementation.
        exclude_info: If True, omit the 'info' block in each embedding result.
        ctx: Authorization context for this request.

    Returns:
        SearchResponse containing search hits for this collection.

    Raises:
        HTTPException: 403 if the caller may not read this collection; 404 if it does not
            exist; 400 if input is invalid.
    """
    collection = await ctx.dal.get_collection_by_name(collection_name)
    if not collection:
        raise HTTPException(status_code=404, detail="collection not found")

    if isinstance(body.input, str):
        raise HTTPException(status_code=400, detail="Raw text search not implemented")

    # numeric vector check
    if not isinstance(body.input, list) or not all(isinstance(x, (int, float)) for x in body.input):
        raise HTTPException(status_code=400, detail="input must be a numeric vector")

    query_vector = body.input
    if len(query_vector) != collection.dimensions:
        raise HTTPException(status_code=400, detail="Input vector dimension mismatch")

    rows = await ctx.dal.search_embeddings_in_collection(
        collection=collection,
        query_vector=query_vector,
        top_k=body.top_k,
        min_value=body.min_value,
        max_value=body.max_value,
        distance_metric=body.distance_metric,
        filters=body.filters,
    )

    results: list[SingleSearchResult] = []
    for row in rows:
        emb = Embedding.from_record(row)
        metric_value = row["value"]
        emb_res = embedding_to_result(emb=emb, collection=collection, exclude_info=exclude_info)
        if isinstance(emb_res, SingleEmbeddingResult):
            results.append(
                SingleSearchResult(
                    id=emb.embedding_id,
                    distance_metric=body.distance_metric,
                    value=metric_value,
                    embedding=emb_res.model_dump(exclude_none=True),
                )
            )

    return SearchResponse(
        embeddings=results,
        collections=[collection_to_model(collection)],
    )


@dual_path(
    vectorstore_search_router,
    "post",
    "/vectorstore/search",
    response_model=SearchResponse,
    summary="Search embeddings across unknown collections",
    description=(
        "Finds the embeddings nearest to the query vector across every collection you have access "
        "to, ordered by similarity. Pass a comma-separated `collections` list to restrict the "
        "search to specific collections. Only collections matching the requested `vector_type` and "
        "the query vector's dimensions are searched; if none of them do, there is nothing the "
        "query could match and the result is empty rather than an error.\n\n"
        f"A single search spans at most {MAX_COLLECTIONS_SEARCHED} collections. If you have access "
        "to more than that, this returns 400 rather than searching a subset, and you search in "
        "batches instead: list your collections with `GET /vectorstore/collections`, then name up "
        f"to {MAX_COLLECTIONS_PER_SEARCH} of them per request in `collections`. Batching does not "
        "change the results. Every result is scored independently of the others, so ordering the "
        "combined batches by `value` - ascending, or descending for `cosine_similarity` - and "
        "keeping the first `top_k` gives exactly what a single search over all of them would."
    ),
    responses={
        **AUTH_RESPONSES,
        **BAD_REQUEST_RESPONSE,
    },
    tags=["Vectorstore Search"],
)
async def search_across_collections(
    body: SearchRequestBody,
    collections: RequestedCollectionNames = None,
    ai_model: AiModel = None,
    vector_type: VectorType = Query(VectorType.vector, alias="vector_type"),
    exclude_info: bool = Query(False, alias="exclude_info"),
    # No collection in the path, so there is no single resource to check: the caller sees
    # whatever their `read` grants make visible, which may be nothing.
    ctx: AuthzContext = Depends(authz("read")),
):
    """
    Perform a vector search across multiple collections.

    Args:
        body: SearchRequestBody containing the query vector and parameters.
        collections: Optional collection names to restrict the search to, parsed and bounded
            from the comma-separated query parameter. None means every collection the caller
            can read, up to MAX_COLLECTIONS_SEARCHED of them.
        ai_model: Optional model name; not used in this minimal implementation.
        vector_type: The type of vector (vector or halfvec) to search against.
        exclude_info: If True, omit the 'info' block in each embedding result.
        ctx: Authorization context for this request.

    Returns:
        SearchResponse containing search hits across collections.

    Raises:
        HTTPException: 400 if invalid collections are specified, if the caller is authorized
            for more collections than one search may span, or if input is invalid.
    """
    if collections is not None:
        collections_list: list[Collection] = []
        for name in collections:
            col = await ctx.dal.get_collection_by_name(name)
            if not col:
                raise HTTPException(status_code=400, detail=f"Invalid collection or unauthorized: {name}")
            collections_list.append(col)
    else:
        # One more than the ceiling, so an over-limit caller is detected rather than served
        # the alphabetically-first page as if it were everything. `list_collections` orders
        # by collection_name, so silently taking its default limit here would search the
        # first N collections by name and return a ranking that looks complete.
        collections_list = await ctx.dal.list_collections(
            limit=MAX_COLLECTIONS_SEARCHED + 1,
        )
        if len(collections_list) > MAX_COLLECTIONS_SEARCHED:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"You are authorized for more than {MAX_COLLECTIONS_SEARCHED} collections, "
                    "which is more than one search can span. Name the collections to search in "
                    f"the `collections` query parameter, at most {MAX_COLLECTIONS_PER_SEARCH} per "
                    "request. Splitting a search this way does not change its results: list your "
                    "collections with GET /vectorstore/collections, search them in batches, then "
                    "order the combined results by `value` (ascending, or descending for "
                    "cosine_similarity) and keep the first `top_k`."
                ),
            )

    if not collections_list:
        return SearchResponse(embeddings=[])

    if isinstance(body.input, str):
        raise HTTPException(status_code=400, detail="Raw text search not implemented")

    if not isinstance(body.input, list) or not all(isinstance(x, (int, float)) for x in body.input):
        raise HTTPException(status_code=400, detail="input must be a numeric vector")

    rows = await ctx.dal.search_embeddings_across_collections(
        collections=collections_list,
        query_vector=body.input,
        top_k=body.top_k,
        min_value=body.min_value,
        max_value=body.max_value,
        distance_metric=body.distance_metric,
        filters=body.filters,
        vector_type=vector_type,
    )

    collection_by_id = {col.id: col for col in collections_list}

    results: list[SingleSearchResult] = []
    hit_collection_ids: set[int] = set()
    for row in rows:
        emb = Embedding.from_record(row)
        col = collection_by_id.get(emb.collection_id)
        if not col:
            # This should never happen if DAL and filtering are consistent.
            # Treat it as a server-side inconsistency.
            raise HTTPException(
                status_code=500,
                detail=f"Search result references unknown collection_id {emb.collection_id}",
            )
        hit_collection_ids.add(col.id)

        metric_value = row["value"]
        emb_res = embedding_to_result(emb, collection=col, exclude_info=exclude_info)
        if isinstance(emb_res, SingleEmbeddingResult):
            results.append(
                SingleSearchResult(
                    id=emb.embedding_id,
                    distance_metric=body.distance_metric,
                    value=metric_value,
                    embedding=emb_res.model_dump(exclude_none=True),
                )
            )

    hit_collections = [collection_to_model(collection_by_id[cid]) for cid in hit_collection_ids]

    return SearchResponse(embeddings=results, collections=hit_collections)
