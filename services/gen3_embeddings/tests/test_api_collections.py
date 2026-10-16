"""Tests for the collection endpoints: create, get, list, update, and delete, and how authz filters each."""

import asyncio
import logging

import asyncpg


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


def _create_index_as_admin(admin_dsn: str, collection_name: str, index_name: str) -> None:
    """Build a vector index on a collection, as the owner role an operator would use."""

    async def _run():
        conn = await asyncpg.connect(admin_dsn)
        try:
            collection_id = await conn.fetchval(
                "SELECT id FROM collections WHERE collection_name = $1", collection_name
            )
            await conn.execute(
                f"CREATE INDEX {index_name} ON embeddings_vector "
                f"USING hnsw ((embedding::vector(3)) vector_cosine_ops) WHERE collection_id = {collection_id}"
            )
        finally:
            await conn.close()

    asyncio.run(_run())


def _drop_index_as_admin(admin_dsn: str, index_name: str) -> None:
    """Drop an index a test built, so it does not outlive the test in the session database."""

    async def _run():
        conn = await asyncpg.connect(admin_dsn)
        try:
            await conn.execute(f"DROP INDEX IF EXISTS {index_name}")
        finally:
            await conn.close()

    asyncio.run(_run())


def _dal_warnings(caplog) -> list[str]:
    """Warnings the DAL's write methods logged, leaving out the auth bypass's warning on every request."""
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING and r.module == "writes"]


def test_delete_collection_warns_about_the_indexes_it_leaves_behind(client, allow_authz, test_database, caplog):
    """The app role cannot drop an index, so deleting an indexed collection names what an operator must drop."""
    allow_authz("alpha", "beta")
    for name in ("alpha", "beta"):
        create_resp = client.post(
            "/vectorstore/collections",
            json={"collection_name": name, "description": "indexed", "dimensions": 3, "vector_type": "vector"},
        )
        assert create_resp.status_code == 200, create_resp.text

    admin_dsn = test_database["admin_dsn"]
    _create_index_as_admin(admin_dsn, "alpha", "idx_delete_warn_alpha")
    _create_index_as_admin(admin_dsn, "beta", "idx_delete_warn_beta")
    try:
        with caplog.at_level(logging.WARNING):
            assert client.delete("/vectorstore/collections/alpha").status_code == 204

        warnings = _dal_warnings(caplog)
        assert len(warnings) == 1
        assert "'alpha'" in warnings[0]
        assert "idx_delete_warn_alpha" in warnings[0]
        assert "idx_delete_warn_beta" not in warnings[0]
    finally:
        _drop_index_as_admin(admin_dsn, "idx_delete_warn_alpha")
        _drop_index_as_admin(admin_dsn, "idx_delete_warn_beta")


def test_delete_collection_without_an_index_does_not_warn(client, allow_authz, caplog):
    """Most collections have no index, and deleting one of those has nothing to report."""
    allow_authz("alpha")
    create_resp = client.post(
        "/vectorstore/collections",
        json={"collection_name": "alpha", "description": "plain", "dimensions": 3, "vector_type": "vector"},
    )
    assert create_resp.status_code == 200, create_resp.text

    with caplog.at_level(logging.WARNING):
        assert client.delete("/vectorstore/collections/alpha").status_code == 204

    assert _dal_warnings(caplog) == []
