import pytest

from gen3_embeddings.database.dal import writes as writes_module
from gen3_embeddings.database.errors import EmbeddingWriteInconsistencyError


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

    page_size = 100  # 3 full pages of `total`, so the walk crosses page boundaries

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
                {"embedding": [0.1, 0.2, 0.3], "metadata": {"source": "good.txt"}},
                {"embedding": [0.1, 0.2], "metadata": {"source": "bad.txt"}},
            ]
        },
    )
    assert response.status_code == 400
    # the DAL's message, which names the offending item by its position in the request
    assert response.json()["detail"] == "Embedding at index 1 has 2 dimensions, expected 3 for this collection"
    assert client.get("/vectorstore/collections/docs/embeddings").json()["embeddings"] == []


def test_update_embedding_dimension_mismatch(client, allow_authz):
    """Updating an embedding to a vector of the wrong length returns 400 and leaves it unchanged."""
    allow_authz("docs")
    client.post(
        "/vectorstore/collections",
        json={"collection_name": "docs", "description": "documents", "dimensions": 3, "vector_type": "vector"},
    )
    embedding_id = client.post(
        "/vectorstore/collections/docs/embeddings",
        json={"embeddings": [{"embedding": [1.0, 0.0, 0.0]}]},
    ).json()["embeddings"][0]["embedding_id"]

    response = client.put(
        f"/vectorstore/collections/docs/embeddings/{embedding_id}",
        json={"embedding": [1.0, 2.0]},
    )
    assert response.status_code == 400, response.text
    assert "has 2 dimensions, expected 3" in response.json()["detail"]
    assert client.get(f"/vectorstore/collections/docs/embeddings/{embedding_id}").json()["vector"] == [1.0, 0.0, 0.0]


def test_upsert_dimension_mismatch_reports_the_request_index_and_writes_nothing(client, allow_authz):
    """
    PUT rejects a wrong-length vector before writing anything.

    Items with an id are updated one transaction at a time before the bulk upsert, so if this
    were left to the DAL the earlier update would already have committed by the time the bad
    vector was found, and the index reported would count only the items without an id.
    """
    allow_authz("docs")
    client.post(
        "/vectorstore/collections",
        json={"collection_name": "docs", "description": "documents", "dimensions": 3, "vector_type": "vector"},
    )
    embedding_id = client.post(
        "/vectorstore/collections/docs/embeddings",
        json={"embeddings": [{"embedding": [1.0, 0.0, 0.0], "metadata": {"v": "old"}}]},
    ).json()["embeddings"][0]["embedding_id"]

    response = client.put(
        "/vectorstore/collections/docs/embeddings",
        json={
            "embeddings": [
                {"embedding_id": embedding_id, "embedding": [0.0, 1.0, 0.0], "metadata": {"v": "new"}},
                {"embedding": [1.0, 2.0]},
            ]
        },
    )
    assert response.status_code == 400, response.text
    assert response.json()["detail"] == "Embedding at index 1 has 2 dimensions, expected 3 for this collection"

    unchanged = client.get(f"/vectorstore/collections/docs/embeddings/{embedding_id}").json()
    assert unchanged["vector"] == [1.0, 0.0, 0.0]
    assert unchanged["info"]["metadata"] == {"v": "old"}


@pytest.mark.parametrize("chunks", [["alpha", "beta", "gamma"], ["1", "2", "3"]])
def test_create_embeddings_rejects_text_chunks(client, allow_authz, chunks):
    """Text chunks validate against the Vector | TextChunks union but are not embeddable yet.

    The numeric-string case matters: pydantic's smart union sends it down the TextChunks arm,
    so it must be refused rather than coerced into a vector.
    """
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

    for method in (client.post, client.put):
        response = method(
            "/vectorstore/collections/docs/embeddings",
            json={"embeddings": [{"embedding": chunks, "metadata": {"source": "notes.txt"}}]},
        )
        assert response.status_code == 400, response.text
        assert "Raw text embedding not implemented" in response.json()["detail"]


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


def _create_three(client) -> list[str]:
    """Create the `docs` collection with three embeddings and return their ids in order."""
    client.post(
        "/vectorstore/collections",
        json={"collection_name": "docs", "description": "documents", "dimensions": 3, "vector_type": "vector"},
    )
    created = client.post(
        "/vectorstore/collections/docs/embeddings",
        json={"embeddings": [{"embedding": [float(i), 0.0, 0.0], "metadata": {"v": "old"}} for i in range(1, 4)]},
    )
    assert created.status_code == 200, created.text
    return [e["embedding_id"] for e in created.json()["embeddings"]]


def test_upsert_updates_every_item_with_an_id_alongside_new_ones(client, allow_authz):
    """Several ids and a new item in one PUT: each id is updated in place, in request order."""
    allow_authz("docs")
    first, second, _ = _create_three(client)

    put_resp = client.put(
        "/vectorstore/collections/docs/embeddings",
        json={
            "embeddings": [
                {"embedding_id": second, "embedding": [0.0, 2.0, 0.0], "metadata": {"v": "new"}},
                {"embedding": [9.0, 9.0, 9.0]},
                {"embedding_id": first, "embedding": [0.0, 1.0, 0.0], "metadata": {"v": "new"}},
            ]
        },
    )
    assert put_resp.status_code == 200, put_resp.text
    results = put_resp.json()["embeddings"]
    assert [r["embedding_id"] for r in (results[0], results[2])] == [second, first]
    assert [r["vector"] for r in results] == [[0.0, 2.0, 0.0], [9.0, 9.0, 9.0], [0.0, 1.0, 0.0]]
    assert client.get(f"/vectorstore/collections/docs/embeddings/{first}").json()["info"]["metadata"] == {"v": "new"}


def test_upsert_with_an_unknown_id_updates_none_of_the_others(client, allow_authz):
    """
    One unknown id fails the request without having written the ids before it.
    """
    allow_authz("docs")
    first, second, _ = _create_three(client)
    unknown = "00000000-0000-4000-8000-000000000000"

    put_resp = client.put(
        "/vectorstore/collections/docs/embeddings",
        json={
            "embeddings": [
                {"embedding_id": first, "embedding": [0.0, 1.0, 0.0], "metadata": {"v": "new"}},
                {"embedding_id": second, "embedding": [0.0, 2.0, 0.0], "metadata": {"v": "new"}},
                {"embedding_id": unknown, "embedding": [0.0, 3.0, 0.0]},
            ]
        },
    )
    assert put_resp.status_code == 400, put_resp.text
    assert unknown in put_resp.json()["detail"]

    for embedding_id, x in ((first, 1.0), (second, 2.0)):
        unchanged = client.get(f"/vectorstore/collections/docs/embeddings/{embedding_id}").json()
        assert unchanged["vector"] == [x, 0.0, 0.0]
        assert unchanged["info"]["metadata"] == {"v": "old"}


def test_upsert_failing_after_the_updates_leaves_nothing_written(client, allow_authz, monkeypatch):
    """
    A failure in the id-less half rolls back the updates by id made earlier in the request.
    """
    allow_authz("docs")
    first, _, _ = _create_three(client)

    def _inconsistent(rows, batch):
        raise EmbeddingWriteInconsistencyError("simulated failure after the INSERT")

    monkeypatch.setattr(writes_module, "_bulk_write_results", _inconsistent)

    put_resp = client.put(
        "/vectorstore/collections/docs/embeddings",
        json={
            "embeddings": [
                {"embedding_id": first, "embedding": [0.0, 1.0, 0.0], "metadata": {"v": "new"}},
                {"embedding": [9.0, 9.0, 9.0]},
            ]
        },
    )
    assert put_resp.status_code == 500, put_resp.text

    unchanged = client.get(f"/vectorstore/collections/docs/embeddings/{first}").json()
    assert unchanged["vector"] == [1.0, 0.0, 0.0]
    assert unchanged["info"]["metadata"] == {"v": "old"}
    assert len(client.get("/vectorstore/collections/docs/embeddings").json()["embeddings"]) == 3


def test_upsert_with_a_repeated_id_is_refused(client, allow_authz):
    """Two items naming the same id would be two writes to one row, so the request is refused."""
    allow_authz("docs")
    first, _, _ = _create_three(client)

    put_resp = client.put(
        "/vectorstore/collections/docs/embeddings",
        json={
            "embeddings": [
                {"embedding_id": first, "embedding": [0.0, 1.0, 0.0]},
                {"embedding_id": first, "embedding": [0.0, 2.0, 0.0]},
            ]
        },
    )
    assert put_resp.status_code == 400, put_resp.text
    assert "more than once" in put_resp.json()["detail"]
    assert client.get(f"/vectorstore/collections/docs/embeddings/{first}").json()["vector"] == [1.0, 0.0, 0.0]


def test_delete_embedding_that_does_not_exist_returns_404(client, allow_authz):
    """
    Deleting an embedding UUID that is not in the collection is a 404.
    """
    allow_authz("docs")

    client.post(
        "/vectorstore/collections",
        json={"collection_name": "docs", "description": "documents", "dimensions": 3, "vector_type": "vector"},
    )

    missing_id = "00000000-0000-0000-0000-000000000000"
    response = client.delete(f"/vectorstore/collections/docs/embeddings/{missing_id}")
    assert response.status_code == 404


def test_delete_embedding_twice_returns_404_the_second_time(client, allow_authz):
    """The first delete removes the embedding; the second has nothing left to remove."""
    allow_authz("docs")

    client.post(
        "/vectorstore/collections",
        json={"collection_name": "docs", "description": "documents", "dimensions": 3, "vector_type": "vector"},
    )
    create_resp = client.post(
        "/vectorstore/collections/docs/embeddings",
        json={"embeddings": [{"embedding": [0.1, 0.2, 0.3], "metadata": {}}]},
    )
    assert create_resp.status_code == 200, create_resp.text
    embedding_id = create_resp.json()["embeddings"][0]["embedding_id"]

    path = f"/vectorstore/collections/docs/embeddings/{embedding_id}"
    assert client.delete(path).status_code == 204
    assert client.delete(path).status_code == 404
