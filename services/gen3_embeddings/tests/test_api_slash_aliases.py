"""
Tests that each endpoint behaves identically with and without a trailing slash.

Every route is registered twice, once at `/path` and once at the undocumented `/path/`.
When those were two separate decorators, the alias silently lost `response_model`,
`response_model_exclude_none` and `status_code`, so `/path/` returned fields the response
model excludes and DELETE aliases answered 200 instead of 204. `dual_path` derives the
alias from the same kwargs; these tests pin the behavior that guarantees.
"""

import pytest

from gen3_embeddings.routes.collections import collections_router
from gen3_embeddings.routes.embeddings import embeddings_router
from gen3_embeddings.routes.embeddings_bulk import embeddings_bulk_router
from gen3_embeddings.routes.search import vectorstore_search_router

ALL_ROUTERS = (
    collections_router,
    embeddings_router,
    embeddings_bulk_router,
    vectorstore_search_router,
)

# attributes that change what a caller actually receives, as opposed to OpenAPI-only ones
BEHAVIORAL_ATTRS = ("response_model", "response_model_exclude_none", "status_code")


def _alias_pairs():
    """Group routes by method and slash-insensitive path, keeping only the registered pairs."""
    grouped: dict[tuple[str, str], list] = {}
    for router in ALL_ROUTERS:
        for route in router.routes:
            for method in route.methods:
                grouped.setdefault((method, route.path.rstrip("/") or "/"), []).append(route)
    return {key: routes for key, routes in grouped.items() if len(routes) > 1}


def test_every_endpoint_registers_a_trailing_slash_alias():
    """Sanity check that there are pairs to compare, so the tests below cannot vacuously pass."""
    pairs = _alias_pairs()
    assert len(pairs) >= 15

    for (method, base), routes in pairs.items():
        paths = sorted(route.path for route in routes)
        assert paths == [base, f"{base}/"], f"unexpected alias shape for {method} {base}: {paths}"


@pytest.mark.parametrize("attr", BEHAVIORAL_ATTRS)
def test_alias_routes_match_canonical_behavior(attr):
    """A `/path/` alias must respond exactly like `/path`, not drop its response contract."""
    mismatched = []
    for (method, base), routes in _alias_pairs().items():
        values = {getattr(route, attr, None) for route in routes}
        if len(values) > 1:
            mismatched.append(f"{method} {base}: {attr}={values}")

    assert not mismatched, "trailing-slash aliases differ from their canonical route:\n" + "\n".join(mismatched)


def test_alias_routes_carry_the_same_dependencies():
    """Auth dependencies must not be dropped from the alias, or it becomes an unguarded route."""
    mismatched = []
    for (method, base), routes in _alias_pairs().items():
        counts = {len(getattr(route, "dependencies", []) or []) for route in routes}
        if len(counts) > 1:
            mismatched.append(f"{method} {base}: dependency counts={counts}")

    assert not mismatched, "trailing-slash aliases have different dependencies:\n" + "\n".join(mismatched)


def test_only_the_canonical_path_is_documented():
    """The alias stays out of the OpenAPI spec so each endpoint appears once."""
    for (method, base), routes in _alias_pairs().items():
        by_path = {route.path: route.include_in_schema for route in routes}
        assert by_path[base] is True, f"{method} {base} should be documented"
        assert by_path[f"{base}/"] is False, f"{method} {base}/ should be hidden"


def test_list_collections_response_is_identical_with_and_without_slash(client, allow_authz):
    """End-to-end: the reported bug, where `/collections/` leaked null fields `/collections` omits."""
    allow_authz("alpha")
    create = client.post(
        "/vectorstore/collections",
        json={"collection_name": "alpha", "description": "a", "dimensions": 3, "vector_type": "vector"},
    )
    assert create.status_code == 200, create.text

    without_slash = client.get("/vectorstore/collections")
    with_slash = client.get("/vectorstore/collections/")

    assert without_slash.status_code == with_slash.status_code
    assert without_slash.json() == with_slash.json()

    # the excluded-None fields specifically must be absent from BOTH
    body = with_slash.json()
    assert "next_page" not in body
    assert "prev_page" not in body
    assert "available_embeddings_count" not in body["collections"][0]


def test_delete_collection_returns_204_with_and_without_slash(client, allow_authz):
    """End-to-end: the DELETE alias used to answer 200 because it lost status_code."""
    allow_authz("alpha")
    create_body = {"collection_name": "alpha", "description": "a", "dimensions": 3, "vector_type": "vector"}

    assert client.post("/vectorstore/collections", json=create_body).status_code == 200
    assert client.delete("/vectorstore/collections/alpha").status_code == 204

    assert client.post("/vectorstore/collections", json=create_body).status_code == 200
    assert client.delete("/vectorstore/collections/alpha/").status_code == 204
