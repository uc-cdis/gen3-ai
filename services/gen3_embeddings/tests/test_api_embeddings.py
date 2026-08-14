import pytest


def test_create_get_and_list_embeddings(client, allow_authz):
    """Creating embeddings returns input_index per item; each can be fetched by ID and appears in list."""
    allow_authz("docs")

    create_collection_resp = client.post(
        "/vectorstore/collections",
        json={
            "collection_name": "docs",
            "description": "documents",
            "dimensions": 3,
            "vector_type": "vector",
        },
    )
    assert create_collection_resp.status_code == 200, create_collection_resp.text

    create_embeddings_resp = client.post(
        "/vectorstore/collections/docs/embeddings",
        json={
            "embeddings": [
                {"embedding": [0.1, 0.2, 0.3], "metadata": {"source": "a.txt"}},
                {"embedding": [0.4, 0.5, 0.6], "metadata": {"source": "b.txt"}},
            ]
        },
    )
    assert create_embeddings_resp.status_code == 200, create_embeddings_resp.text

    created = create_embeddings_resp.json()["embeddings"]
    assert len(created) == 2
    assert created[0]["input_index"] == 0
    assert created[1]["input_index"] == 1

    embedding_id = created[0]["embedding_id"]

    get_resp = client.get(f"/vectorstore/collections/docs/embeddings/{embedding_id}")
    assert get_resp.status_code == 200, get_resp.text

    emb = get_resp.json()
    assert emb["embedding_id"] == embedding_id
    # "vector" columns store float32, so 0.1/0.2/0.3 round-trip lossily
    assert emb["vector"] == pytest.approx([0.1, 0.2, 0.3], rel=1e-6)
    assert emb["info"]["authz"] == "/vectorstore/collections/docs"

    list_resp = client.get("/vectorstore/collections/docs/embeddings")
    assert list_resp.status_code == 200, list_resp.text

    listed = list_resp.json()
    assert len(listed["embeddings"]) == 2


def test_list_embeddings_in_collection_pagination_is_consistent_across_pages(client, allow_authz):
    """Walking every page of a collection's embeddings returns each one exactly once, in the same order, run after run."""
    allow_authz("docs")

    create_collection_resp = client.post(
        "/vectorstore/collections",
        json={"collection_name": "docs", "description": "documents", "dimensions": 3, "vector_type": "vector"},
    )
    assert create_collection_resp.status_code == 200, create_collection_resp.text

    total = 300
    create_resp = client.post(
        "/vectorstore/collections/docs/embeddings",
        json={"embeddings": [{"embedding": [0.0, 0.0, 0.0], "metadata": {"i": i}} for i in range(total)]},
    )
    assert create_resp.status_code == 200, create_resp.text
    expected_ids = {e["embedding_id"] for e in create_resp.json()["embeddings"]}
    assert len(expected_ids) == total

    page_size = 100  # minimum page_size this endpoint allows (DEFAULT_PAGE_SIZE)

    def walk_all_pages() -> list[str]:
        seen_ids: list[str] = []
        page = 1
        pages_fetched = 0

        while page is not None:
            resp = client.get(
                "/vectorstore/collections/docs/embeddings",
                params={"page": page, "page_size": page_size},
            )
            assert resp.status_code == 200, resp.text
            data = resp.json()

            seen_ids.extend(e["embedding_id"] for e in data["embeddings"])

            pages_fetched += 1
            # guard against an infinite loop if next_page never becomes None
            assert pages_fetched <= total

            page = data["next_page"]

        return seen_ids

    first_walk = walk_all_pages()

    # every embedding is returned exactly once across all pages: no gaps, no duplicates
    assert len(first_walk) == total
    assert set(first_walk) == expected_ids
    assert len(first_walk) == len(set(first_walk))

    # pagination must be deterministic: repeated full walks return the exact same order every time
    for _ in range(9):
        assert walk_all_pages() == first_walk


def test_create_embeddings_dimension_mismatch(client, allow_authz):
    """Posting a vector whose length differs from the collection's dimensions returns 400."""
    allow_authz("docs")

    create_collection_resp = client.post(
        "/vectorstore/collections",
        json={
            "collection_name": "docs",
            "description": "documents",
            "dimensions": 3,
            "vector_type": "vector",
        },
    )
    assert create_collection_resp.status_code == 200, create_collection_resp.text

    response = client.post(
        "/vectorstore/collections/docs/embeddings",
        json={
            "embeddings": [
                {"embedding": [0.1, 0.2], "metadata": {"source": "bad.txt"}},
            ]
        },
    )
    assert response.status_code == 400
    assert "Embedding dimension mismatch" in response.json()["detail"]


def test_update_embedding(client, allow_authz):
    """PUT on a single embedding by ID replaces its vector and metadata."""
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
                {"embedding": [1.0, 2.0, 3.0], "metadata": {"tag": "before"}},
            ]
        },
    )
    assert create_resp.status_code == 200, create_resp.text

    embedding_id = create_resp.json()["embeddings"][0]["embedding_id"]

    update_resp = client.put(
        f"/vectorstore/collections/docs/embeddings/{embedding_id}",
        json={
            "embedding": [9.0, 8.0, 7.0],
            "metadata": {"tag": "after"},
        },
    )
    assert update_resp.status_code == 200, update_resp.text

    updated = update_resp.json()
    assert updated["vector"] == [9.0, 8.0, 7.0]
    assert updated["info"]["metadata"] == {"tag": "after"}


def test_delete_embedding(client, allow_authz):
    """Deleting an embedding returns 204, and subsequent GET returns 404."""
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
                {"embedding": [0.1, 0.2, 0.3], "metadata": {}},
            ]
        },
    )
    assert create_resp.status_code == 200, create_resp.text

    embedding_id = create_resp.json()["embeddings"][0]["embedding_id"]

    delete_resp = client.delete(f"/vectorstore/collections/docs/embeddings/{embedding_id}")
    assert delete_resp.status_code == 204

    get_resp = client.get(f"/vectorstore/collections/docs/embeddings/{embedding_id}")
    assert get_resp.status_code == 404


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


def test_upsert_embeddings(client, allow_authz):
    """PUT with an existing embedding_id replaces that embedding's vector and metadata in place."""
    allow_authz("docs")

    client.post(
        "/vectorstore/collections",
        json={"collection_name": "docs", "description": "documents", "dimensions": 3, "vector_type": "vector"},
    )

    create_resp = client.post(
        "/vectorstore/collections/docs/embeddings",
        json={"embeddings": [{"embedding": [1.0, 2.0, 3.0], "metadata": {"v": "1"}}]},
    )
    assert create_resp.status_code == 200
    embedding_id = create_resp.json()["embeddings"][0]["embedding_id"]

    put_resp = client.put(
        "/vectorstore/collections/docs/embeddings",
        json={"embeddings": [{"embedding": [4.0, 5.0, 6.0], "metadata": {"v": "2"}, "embedding_id": embedding_id}]},
    )
    assert put_resp.status_code == 200, put_resp.text
    result = put_resp.json()["embeddings"][0]
    assert result["embedding_id"] == embedding_id
    assert result["vector"] == [4.0, 5.0, 6.0]
    assert result["info"]["metadata"] == {"v": "2"}


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
