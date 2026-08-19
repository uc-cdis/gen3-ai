"""Tests for the vector search endpoints (routes/search.py)."""


def test_search_embeddings_in_collection(client, allow_authz):
    """Search within one collection returns the nearest vector with the requested distance metric."""
    allow_authz("docs")

    client.post(
        "/vectorstore/collections",
        json={
            "collection_name": "docs",
            "description": "documents",
            "dimensions": 3,
            "vector_type": "vector",
        },
    )

    create_resp = client.post(
        "/vectorstore/collections/docs/embeddings",
        json={
            "embeddings": [
                {"embedding": [1.0, 0.0, 0.0], "metadata": {"label": "x"}},
                {"embedding": [0.0, 1.0, 0.0], "metadata": {"label": "y"}},
            ]
        },
    )
    assert create_resp.status_code == 200, create_resp.text

    search_resp = client.post(
        "/vectorstore/collections/docs/search",
        json={
            "input": [1.0, 0.0, 0.0],
            "top_k": 1,
            "distance_metric": "cosine_similarity",
        },
    )
    assert search_resp.status_code == 200, search_resp.text

    data = search_resp.json()
    assert len(data["embeddings"]) == 1
    assert data["embeddings"][0]["distance_metric"] == "cosine_similarity"
    assert data["collections"][0]["collection_name"] == "docs"


def test_search_across_collections(client, allow_authz):
    """Global search returns hits from all accessible collections and reports which collections matched."""
    allow_authz("coll_a", "coll_b")

    for name in ("coll_a", "coll_b"):
        client.post(
            "/vectorstore/collections",
            json={"collection_name": name, "description": name, "dimensions": 3, "vector_type": "vector"},
        )

    client.post(
        "/vectorstore/collections/coll_a/embeddings",
        json={"embeddings": [{"embedding": [1.0, 0.0, 0.0], "metadata": {"coll": "a"}}]},
    )
    client.post(
        "/vectorstore/collections/coll_b/embeddings",
        json={"embeddings": [{"embedding": [0.0, 1.0, 0.0], "metadata": {"coll": "b"}}]},
    )

    search_resp = client.post(
        "/vectorstore/search",
        json={"input": [1.0, 0.0, 0.0], "top_k": 5, "distance_metric": "cosine_similarity"},
    )
    assert search_resp.status_code == 200, search_resp.text

    data = search_resp.json()
    assert len(data["embeddings"]) > 0
    collection_names = {c["collection_name"] for c in data["collections"]}
    assert "coll_a" in collection_names
