def test_list_collections_empty(client, allow_authz):
    """An authorized user with no collections yet gets an empty list on page 1."""
    allow_authz("alpha", "beta")

    response = client.get("/vectorstore/collections")
    assert response.status_code == 200

    data = response.json()
    assert data["collections"] == []
    assert data["page"] == 1


def test_create_and_get_collection(client, allow_authz):
    """Creating a collection returns the full model, and it can be fetched by name."""
    allow_authz("alpha")

    create_resp = client.post(
        "/vectorstore/collections",
        json={
            "collection_name": "alpha",
            "description": "test collection",
            "dimensions": 3,
            "vector_type": "vector",
        },
    )
    assert create_resp.status_code == 200, create_resp.text

    created = create_resp.json()
    assert created["collection_name"] == "alpha"
    assert created["description"] == "test collection"
    assert created["dimensions"] == 3
    assert created["vector_type"] == "vector"
    assert created["self"] == "/vectorstore/collections/alpha"

    get_resp = client.get("/vectorstore/collections/alpha")
    assert get_resp.status_code == 200, get_resp.text

    fetched = get_resp.json()
    assert fetched["collection_name"] == "alpha"
    assert fetched["dimensions"] == 3


def test_create_collection_unauthorized(client, allow_authz):
    """Creating a collection the caller has no authz for returns 403."""
    allow_authz("other_collection")

    response = client.post(
        "/vectorstore/collections",
        json={
            "collection_name": "alpha",
            "description": "should fail",
            "dimensions": 3,
            "vector_type": "vector",
        },
    )
    assert response.status_code == 403


def test_list_collections_filters_by_allowed_authz(client, allow_authz):
    """List only returns collections whose authz path the caller is allowed to see."""
    allow_authz("alpha", "beta")

    resp1 = client.post(
        "/vectorstore/collections",
        json={
            "collection_name": "alpha",
            "description": "a",
            "dimensions": 3,
            "vector_type": "vector",
        },
    )
    assert resp1.status_code == 200, resp1.text

    resp2 = client.post(
        "/vectorstore/collections",
        json={
            "collection_name": "beta",
            "description": "b",
            "dimensions": 4,
            "vector_type": "halfvec",
        },
    )
    assert resp2.status_code == 200, resp2.text

    # Narrow authz for the list call
    allow_authz("alpha")

    list_resp = client.get("/vectorstore/collections")
    assert list_resp.status_code == 200, list_resp.text

    data = list_resp.json()
    names = [c["collection_name"] for c in data["collections"]]
    assert names == ["alpha"]


def test_list_collections_pagination_is_consistent_across_pages(client, allow_authz):
    """Walking every page returns each collection exactly once, in the same order, run after run."""
    names = [f"coll_{i}" for i in range(100)]
    allow_authz(*names)

    for name in names:
        resp = client.post(
            "/vectorstore/collections",
            json={"collection_name": name, "description": name, "dimensions": 3, "vector_type": "vector"},
        )
        assert resp.status_code == 200, resp.text

    page_size = 10

    def walk_all_pages() -> list[str]:
        seen_names: list[str] = []
        page = 1
        pages_fetched = 0

        while page is not None:
            resp = client.get("/vectorstore/collections", params={"page": page, "page_size": page_size})
            assert resp.status_code == 200, resp.text
            data = resp.json()

            seen_names.extend(c["collection_name"] for c in data["collections"])

            pages_fetched += 1
            # guard against an infinite loop if next_page never becomes None
            assert pages_fetched <= len(names)

            # next_page is omitted from the JSON body (not null) once it's None,
            # because the route uses response_model_exclude_none=True
            page = data.get("next_page")

        return seen_names

    first_walk = walk_all_pages()

    # every collection is returned exactly once across all pages: no gaps, no duplicates
    assert len(first_walk) == len(names)
    assert sorted(first_walk) == sorted(names)
    assert len(first_walk) == len(set(first_walk))

    # pagination must be deterministic: repeated full walks return the exact same order every time
    for _ in range(9):
        assert walk_all_pages() == first_walk


def test_get_collection_not_found_when_not_allowed(client, allow_authz):
    """Fetching a collection the caller has no authz for returns 404"""
    allow_authz("beta")

    client.post(
        "/vectorstore/collections",
        json={
            "collection_name": "beta",
            "description": "b",
            "dimensions": 3,
            "vector_type": "vector",
        },
    )

    response = client.get("/vectorstore/collections/alpha")
    assert response.status_code == 404
    assert response.json()["detail"] == "Collection not found"


def test_update_collection(client, allow_authz):
    """PATCH on an existing collection updates only the description and returns the full model."""
    allow_authz("alpha")

    create_resp = client.post(
        "/vectorstore/collections",
        json={
            "collection_name": "alpha",
            "description": "before",
            "dimensions": 3,
            "vector_type": "vector",
        },
    )
    assert create_resp.status_code == 200, create_resp.text

    patch_resp = client.patch(
        "/vectorstore/collections/alpha",
        json={"description": "after"},
    )
    assert patch_resp.status_code == 200, patch_resp.text

    updated = patch_resp.json()
    assert updated["collection_name"] == "alpha"
    assert updated["description"] == "after"


def test_delete_collection(client, allow_authz):
    """Deleting a collection returns 204, and subsequent GET returns 404."""
    allow_authz("alpha")

    create_resp = client.post(
        "/vectorstore/collections",
        json={
            "collection_name": "alpha",
            "description": "to delete",
            "dimensions": 3,
            "vector_type": "vector",
        },
    )
    assert create_resp.status_code == 200, create_resp.text

    delete_resp = client.delete("/vectorstore/collections/alpha")
    assert delete_resp.status_code == 204

    get_resp = client.get("/vectorstore/collections/alpha")
    assert get_resp.status_code == 404


def test_delete_collection_that_does_not_exist_returns_404(client, allow_authz):
    """
    Deleting a name that was never created.
    """
    allow_authz("alpha")

    response = client.delete("/vectorstore/collections/alpha")
    assert response.status_code == 404


def test_delete_collection_twice_returns_404_the_second_time(client, allow_authz):
    """The first delete removes the collection; the second has nothing left to remove."""
    allow_authz("alpha")

    create_resp = client.post(
        "/vectorstore/collections",
        json={"collection_name": "alpha", "description": "to delete", "dimensions": 3, "vector_type": "vector"},
    )
    assert create_resp.status_code == 200, create_resp.text

    assert client.delete("/vectorstore/collections/alpha").status_code == 204
    assert client.delete("/vectorstore/collections/alpha").status_code == 404
