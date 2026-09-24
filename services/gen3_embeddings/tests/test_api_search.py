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


def test_search_across_collections_none_of_which_match_the_query_is_empty(client, allow_authz):
    """A query no collection could match returns no hits rather than failing."""
    allow_authz("dim3")
    client.post(
        "/vectorstore/collections",
        json={"collection_name": "dim3", "description": "dim3", "dimensions": 3, "vector_type": "vector"},
    )
    client.post(
        "/vectorstore/collections/dim3/embeddings",
        json={"embeddings": [{"embedding": [1.0, 0.0, 0.0]}]},
    )

    # Four dimensions against a collection of three, so the widths cannot match.
    query = {"input": [1.0, 0.0, 0.0, 0.0], "top_k": 5}

    named = client.post("/vectorstore/search", params={"collections": "dim3"}, json=query)
    assert named.status_code == 200, named.text
    assert named.json()["embeddings"] == []

    unnamed = client.post("/vectorstore/search", json=query)
    assert unnamed.status_code == 200, unnamed.text
    assert unnamed.json()["embeddings"] == []


def test_search_in_batches_matches_one_search_over_every_collection(client, allow_authz):
    """
    Splitting a search across batches of collections gives the same results as not splitting.

    This is the property that makes the collection ceiling on cross-collection search an
    inconvenience rather than a loss of results: a caller over the bound names collections
    in batches and merges the responses. It holds because every hit is scored by a metric
    over that row alone, with no per-collection quota and no normalization across the set,
    so the nearest `top_k` of the whole is the nearest `top_k` of the batches' `top_k`s.
    """
    names = ("batch_a", "batch_b", "batch_c", "batch_d")
    allow_authz(*names)

    for i, name in enumerate(names):
        client.post(
            "/vectorstore/collections",
            json={"collection_name": name, "description": name, "dimensions": 3, "vector_type": "vector"},
        )
        # Every vector tilts off the query axis by a different amount, so all eight scores
        # are distinct and the expected order is total rather than a tie the two searches
        # could break differently. The offsets interleave the collections, which is what
        # makes merging the batches worth checking at all.
        client.post(
            f"/vectorstore/collections/{name}/embeddings",
            json={
                "embeddings": [
                    {"embedding": [1.0, (j * len(names) + i + 1) / 20.0, 0.0], "metadata": {"coll": name}}
                    for j in range(2)
                ]
            },
        )

    query = {"input": [1.0, 0.0, 0.0], "top_k": 3, "distance_metric": "cosine_similarity"}

    unbatched = client.post("/vectorstore/search", json=query)
    assert unbatched.status_code == 200, unbatched.text

    merged: list[dict] = []
    for batch in (names[:2], names[2:]):
        batched = client.post("/vectorstore/search", params={"collections": ",".join(batch)}, json=query)
        assert batched.status_code == 200, batched.text
        merged.extend(batched.json()["embeddings"])

    # cosine_similarity is the one metric where a larger value is closer, hence reverse.
    merged.sort(key=lambda hit: hit["value"], reverse=True)
    merged = merged[: query["top_k"]]

    expected = unbatched.json()["embeddings"]
    # Guards against the comparison below passing on two empty lists.
    assert len(expected) == query["top_k"]
    assert [hit["id"] for hit in merged] == [hit["id"] for hit in expected]
    # The top hits have to come from both batches, or merging them proved nothing.
    assert len({hit["embedding"]["info"]["metadata"]["coll"] for hit in expected}) > 1
