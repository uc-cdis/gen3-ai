from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from gen3_embeddings.db import DataAccessLayer, Embedding, VectorIndex, get_embedding_dal
from gen3_embeddings.models.helpers import embedding_to_single_result, vector_index_to_model
from gen3_embeddings.models.schemas import (
    SearchRequest,
    SearchResponse,
    SearchResponseNoIndex,
    SingleEmbeddingResult,
    SingleEmbeddingResultNoIndex,
    SingleSearchResult,
    SingleSearchResultNoIndex,
)

vector_search_router = APIRouter(tags=["Vector Search"])


@vector_search_router.post(
    "/vector/indices/{index_name}/search",
    response_model=SearchResponseNoIndex,
    summary="Search embeddings in index",
)
async def search_in_index(
    index_name: str,
    request: SearchRequest,
    ai_model: str | None = Query(None, alias="ai_model"),
    no_embeddings_info: bool = Query(False, alias="no_embeddings_info"),
    dal: DataAccessLayer = Depends(get_embedding_dal),
):
    """
    TODO: support for ai_model
    TODO: raw text search

    Perform a vector search within a specific index.

    Args:
        index_name: Name of the vector index to search.
        request: SearchRequest containing the query vector and parameters.
        ai_model: Optional model name; not used in this minimal implementation.
        no_embeddings_info: If True, omit the 'info' block in each embedding result.
        dal: Data access layer dependency.

    Returns:
        SearchResponseNoIndex containing search hits for this index.

    Raises:
        HTTPException: 404 if index is not found; 400 if input is invalid.
    """
    index = await dal.get_vector_index_by_name(index_name)
    if not index:
        raise HTTPException(status_code=404, detail="Index not found")

    if isinstance(request.input, str):
        raise HTTPException(status_code=400, detail="Raw text search not implemented")

    query_vector = request.input
    if len(query_vector) != index.dimensions:
        raise HTTPException(status_code=400, detail="Input vector dimension mismatch")

    rows = await dal.search_embeddings_in_index(
        vector_index_id=index.id,
        query_vector=query_vector,
        top_k=request.top_k,
        score_range=request.range,
        filters=request.filters,
    )

    results: list[SingleSearchResultNoIndex] = []
    for row in rows:
        emb = Embedding.from_record(row)
        sim = row["similarity_score"]
        emb_res = embedding_to_single_result(emb=emb, index=index, include_index=False)
        if isinstance(emb_res, SingleEmbeddingResultNoIndex):
            if no_embeddings_info:
                emb_res.info = None
            results.append(
                SingleSearchResultNoIndex(
                    id=emb.embedding_id,
                    similarity_score=sim,
                    embedding=emb_res,
                )
            )

    return SearchResponseNoIndex(
        embeddings=results,
        vector_indices=[vector_index_to_model(index)],
    )


@vector_search_router.post(
    "/vector/search",
    response_model=SearchResponse,
    summary="Search embeddings across unknown indices",
)
async def search_across_indices(
    request: SearchRequest,
    vector_indices: str | None = Query(None, alias="vector_indices"),
    ai_model: str | None = Query(None, alias="ai_model"),
    no_embeddings_info: bool = Query(False, alias="no_embeddings_info"),
    dal: DataAccessLayer = Depends(get_embedding_dal),
):
    """
    TODO: support for ai_model
    TODO: how to handle diffs in dimensions? current logic is not sufficient.

    Perform a vector search across multiple indices.

    Args:
        request: SearchRequest containing the query vector and parameters.
        vector_indices: Optional comma-separated list of index names to restrict the search.
        ai_model: Optional model name; not used in this minimal implementation.
        no_embeddings_info: If True, omit the 'info' block in each embedding result.
        dal: Data access layer dependency.

    Returns:
        SearchResponse containing search hits across indices.

    Raises:
        HTTPException: 400 if invalid indices are specified or input is invalid.
    """
    if vector_indices:
        names = [v.strip() for v in vector_indices.split(",") if v.strip()]
        indices: list[VectorIndex] = []
        for name in names:
            idx = await dal.get_vector_index_by_name(name)
            if not idx:
                raise HTTPException(status_code=400, detail=f"Invalid vector index: {name}")
            indices.append(idx)
    else:
        indices = await dal.list_vector_indices()

    if not indices:
        return SearchResponse(embeddings=[])

    if isinstance(request.input, str):
        raise HTTPException(status_code=400, detail="Raw text search not implemented")

    dims = indices[0].dimensions
    if len(request.input) != dims:
        raise HTTPException(status_code=400, detail="Input vector dimension mismatch")

    index_ids = [idx.id for idx in indices]
    rows = await dal.search_embeddings_across_indices(
        vector_index_ids=index_ids,
        query_vector=request.input,
        top_k=request.top_k,
        score_range=request.range,
        filters=request.filters,
    )

    index_by_id = {idx.id: idx for idx in indices}

    results: list[SingleSearchResult] = []
    for row in rows:
        emb = Embedding.from_record(row)
        idx = index_by_id.get(emb.vector_index_id)
        if not idx:
            continue
        sim = row["similarity_score"]
        emb_res = embedding_to_single_result(emb, index=idx, include_index=True)
        if isinstance(emb_res, SingleEmbeddingResult):
            if no_embeddings_info:
                emb_res.info = None
            results.append(
                SingleSearchResult(
                    id=emb.embedding_id,
                    similarity_score=sim,
                    embedding=emb_res,
                )
            )

    return SearchResponse(embeddings=results)
