"""
Tests that each route authorizes the action it actually performs.

The action used to be derived from the HTTP verb, which is wrong for the endpoints that read
but are declared POST because their query does not fit in a query string. Those routes now
declare `authz("read")`, and these tests pin that: a caller holding only `read` must be able
to use them, and a caller holding only `create` must not.

They also pin the structural property that makes the whole thing traceable -- every
non-public route carries exactly one authz declaration, so a new route cannot be added
unguarded without this failing.

DEBUG_SKIP_AUTH is on for tests, so the Arborist call inside the dependency is a no-op. What
these exercise is the other half: which resource set the request runs under, and therefore
what row-level security lets through. See test_collection_name_normalization.py for the
resource path the Arborist call would receive.
"""

import inspect

import pytest
from fastapi.routing import APIRoute

from gen3_embeddings.dependencies import AuthzDependency, authz
from gen3_embeddings.routes.collections import collections_router
from gen3_embeddings.routes.embeddings import embeddings_router
from gen3_embeddings.routes.embeddings_bulk import embeddings_bulk_router
from gen3_embeddings.routes.search import vectorstore_search_router

# Every router carrying data endpoints. `basic_router` is deliberately absent: its routes
# (`/`, `/_version`, `/_status`) are public by design.
DATA_ROUTERS = (
    collections_router,
    embeddings_router,
    embeddings_bulk_router,
    vectorstore_search_router,
)

COLLECTION = {"collection_name": "docs", "description": "d", "dimensions": 3, "vector_type": "vector"}


def _authz_dependencies(route: APIRoute) -> list[AuthzDependency]:
    """Collect the `authz(...)` declarations on a route."""
    return [dep.call for dep in route.dependant.dependencies if isinstance(dep.call, AuthzDependency)]


def _authz_declarations(route: APIRoute) -> list[str]:
    """Collect the actions declared by `authz(...)` dependencies on a route."""
    return [dep.action for dep in _authz_dependencies(route)]


def _data_routes() -> list[APIRoute]:
    """Every data endpoint, including the undocumented trailing-slash aliases."""
    return [route for router in DATA_ROUTERS for route in router.routes if isinstance(route, APIRoute)]


def test_every_route_declares_exactly_one_authz_action():
    """
    An endpoint with no authz declaration is an unauthorized endpoint.

    Two declarations would be just as bad: the second would silently overwrite the first
    handler argument, so only one of them would scope the DAL.
    """
    routes = _data_routes()
    # so this cannot pass by finding no routes at all
    assert len(routes) >= 30

    wrong = {
        f"{sorted(route.methods)} {route.path}": _authz_declarations(route)
        for route in routes
        if len(_authz_declarations(route)) != 1
    }
    assert not wrong, f"routes without exactly one authz(...) declaration: {wrong}"


def test_declared_actions_match_what_each_route_does():
    """
    The declared action per endpoint, pinned as a table.

    This is the artifact the refactor exists to produce: the authorization for every
    endpoint, readable in one place. A deliberate change should update this table.
    """
    declared = {
        (tuple(sorted(route.methods)), route.path): _authz_declarations(route)[0]
        for route in _data_routes()
        if route.include_in_schema
    }

    assert declared == {
        (("GET",), "/vectorstore/collections"): "read",
        (("POST",), "/vectorstore/collections"): "create",
        (("GET",), "/vectorstore/collections/{collection_name}"): "read",
        (("PATCH",), "/vectorstore/collections/{collection_name}"): "update",
        (("DELETE",), "/vectorstore/collections/{collection_name}"): "delete",
        (("GET",), "/vectorstore/collections/{collection_name}/embeddings"): "read",
        (("POST",), "/vectorstore/collections/{collection_name}/embeddings"): "create",
        (("PUT",), "/vectorstore/collections/{collection_name}/embeddings"): "update",
        (("GET",), "/vectorstore/collections/{collection_name}/embeddings/{embedding_uuid}"): "read",
        (("PUT",), "/vectorstore/collections/{collection_name}/embeddings/{embedding_uuid}"): "update",
        (("DELETE",), "/vectorstore/collections/{collection_name}/embeddings/{embedding_uuid}"): "delete",
        # POST, but reads. These are the ones the verb got wrong.
        (("POST",), "/embeddings/bulk"): "read",
        (("POST",), "/vectorstore/collections/{collection_name}/embeddings/bulk"): "read",
        (("POST",), "/vectorstore/collections/{collection_name}/search"): "read",
        (("POST",), "/vectorstore/search"): "read",
    }


def test_upsert_route_also_requires_create():
    """
    PUT on the embeddings collection both creates and updates, so it demands both.

    `update` stays the primary action, since that is what scopes row-level security.
    """
    route = next(
        route
        for route in _data_routes()
        if route.path == "/vectorstore/collections/{collection_name}/embeddings" and "PUT" in route.methods
    )
    (dep,) = _authz_dependencies(route)
    assert dep.action == "update"
    assert dep.also_require == ("create",)


def test_alias_routes_declare_the_same_action():
    """A trailing-slash alias that lost its authz declaration would be an open endpoint."""
    by_key: dict[tuple, set[str]] = {}
    for route in _data_routes():
        key = (tuple(sorted(route.methods)), route.path.rstrip("/") or "/")
        by_key.setdefault(key, set()).update(_authz_declarations(route))

    inconsistent = {key: actions for key, actions in by_key.items() if len(actions) != 1}
    assert not inconsistent, f"path and its trailing-slash alias disagree: {inconsistent}"


# ---------------------------------------------------------------------------
# Behavioral: a read-only caller can use the POST-shaped read endpoints
# ---------------------------------------------------------------------------


@pytest.fixture
def seeded_docs(client, allow_authz):
    """Create the `docs` collection with one embedding, then return its embedding UUID."""
    allow_authz("docs")
    assert client.post("/vectorstore/collections", json=COLLECTION).status_code == 200

    created = client.post(
        "/vectorstore/collections/docs/embeddings",
        json={"embeddings": [{"embedding": [1.0, 0.0, 0.0], "metadata": {"n": "1"}}]},
    )
    assert created.status_code == 200, created.text
    return created.json()["embeddings"][0]["embedding_id"]


@pytest.mark.parametrize(
    "method,path,body",
    [
        ("post", "/vectorstore/collections/docs/search", {"input": [1.0, 0.0, 0.0], "top_k": 5}),
        ("post", "/vectorstore/search", {"input": [1.0, 0.0, 0.0], "top_k": 5}),
    ],
)
def test_read_only_caller_can_use_post_search(client, allow_authz_per_action, seeded_docs, method, path, body):
    """
    A caller granted only `read` can search, even though search is a POST.

    Under verb-derived authorization this ran as `create`, so a read-only caller got an empty
    result from an endpoint they were entitled to use.
    """
    allow_authz_per_action(read=("docs",), create=(), update=(), delete=())

    response = getattr(client, method)(path, json=body)
    assert response.status_code == 200, response.text
    assert len(response.json()["embeddings"]) == 1


def test_read_only_caller_can_use_post_bulk_read(client, allow_authz_per_action, seeded_docs):
    """Same for the bulk read endpoints, which are POST for the same reason."""
    allow_authz_per_action(read=("docs",), create=(), update=(), delete=())

    scoped = client.post("/vectorstore/collections/docs/embeddings/bulk", json=[seeded_docs])
    assert scoped.status_code == 200, scoped.text
    assert len(scoped.json()["embeddings"]) == 1

    unknown = client.post("/embeddings/bulk", json=[seeded_docs])
    assert unknown.status_code == 200, unknown.text
    assert len(unknown.json()["embeddings"]) == 1


def test_create_only_caller_cannot_read(client, allow_authz_per_action, seeded_docs):
    """
    The converse: holding `create` does not let a caller read.

    Without this, "declare the action" would be satisfied by a dependency that ignored the
    action and returned every grant.
    """
    allow_authz_per_action(read=(), create=("docs",), update=(), delete=())

    assert client.get("/vectorstore/collections/docs").status_code == 404
    assert client.get("/vectorstore/collections").json()["collections"] == []
    assert client.post("/vectorstore/collections/docs/search", json={"input": [1.0, 0.0, 0.0]}).status_code == 404
    assert client.post("/vectorstore/search", json={"input": [1.0, 0.0, 0.0]}).json()["embeddings"] == []


def test_delete_grant_alone_does_not_authorize_writes(client, allow_authz_per_action, seeded_docs):
    """A `delete`-only caller cannot create embeddings, and cannot read the collection."""
    allow_authz_per_action(read=(), create=(), update=(), delete=("docs",))

    created = client.post(
        "/vectorstore/collections/docs/embeddings",
        json={"embeddings": [{"embedding": [0.0, 1.0, 0.0]}]},
    )
    assert created.status_code == 404, created.text


def test_authz_dependency_signature_takes_only_the_request(app):
    """
    The dependency depends on nothing but the request.

    It reads `collection_name` from `request.path_params` rather than declaring it, so it
    cannot be reordered relative to the parameter parsing it would otherwise depend on, and
    one declaration works for every route whatever else that route's signature holds.
    """
    dependency = authz("read")
    assert list(inspect.signature(dependency).parameters) == ["request"]
