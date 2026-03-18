from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from gen3_embeddings.db import get_embedding_dal

embeddings_router = APIRouter()


@embeddings_router.get("/embeddings/{embedding_uuid}")
async def get_embedding(embedding_uuid: UUID, dal=Depends(get_embedding_dal)):
    emb = await dal.get_embedding_by_id(embedding_uuid)
    if not emb:
        raise HTTPException(status_code=404, detail="Embedding not found")
    return emb


@embeddings_router.get("/vector/indices/{index_name}/embeddings/{embedding_uuid}")
async def get_embedding_from_index(index_name: str, embedding_uuid: UUID, dal=Depends(get_embedding_dal)):
    index = await dal.get_vector_index(index_name)
    if not index:
        raise HTTPException(status_code=404, detail="Index not found")
    emb = await dal.get_embedding_by_index_and_id(index["id"], embedding_uuid)
    if not emb:
        raise HTTPException(status_code=404, detail="Embedding not found")
    return emb


@embeddings_router.put("/vector/indices/{index_name}/embeddings/{embedding_uuid}")
async def update_embedding_in_index(
    index_name: str, embedding_uuid: UUID, embedding: list[float], dal=Depends(get_embedding_dal)
):
    index = await dal.get_vector_index(index_name)
    if not index:
        raise HTTPException(status_code=404, detail="Index not found")
    emb = await dal.update_embedding(index["id"], embedding_uuid, embedding)
    if not emb:
        raise HTTPException(status_code=400, detail="Failed to update embedding")
    return emb


@embeddings_router.delete("/vector/indices/{index_name}/embeddings/{embedding_uuid}")
async def delete_embedding(index_name: str, embedding_uuid: UUID, dal=Depends(get_embedding_dal)):
    index = await dal.get_vector_index(index_name)
    if not index:
        raise HTTPException(status_code=404, detail="Index not found")
    result = await dal.delete_embedding(index["id"], embedding_uuid)
    if not result:
        raise HTTPException(status_code=404, detail="Embedding not found or already deleted")
    # status_code=204 implied
    return None
