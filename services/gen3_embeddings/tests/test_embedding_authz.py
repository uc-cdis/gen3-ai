"""
Tests that an embedding's `authz` is an arbitrary, mutable value.

Two things are easy to conflate and behave differently:

- A COLLECTION's authz resource is derived from its name and cannot be anything else.
- An EMBEDDING's authz is a stored string. It defaults to its collection's path, but a
  caller may set it to any resource they hold the action on, and may change it later.

The collection convention is enforced in `auth.get_allowed_collection_names_from_authz`,
which narrows a caller's grants to collection-shaped paths. These tests pin that the
narrowing does NOT leak into the embeddings path: the RLS setting the embeddings tables read
is the caller's grants unnarrowed, so a row can carry an authz path this service has no
convention for.

DEBUG_SKIP_AUTH is on for tests, so the policy-engine call is a no-op. What is exercised here
is the resulting RLS context, which is the half that actually stores and retrieves the value.
"""

import pytest

COLLECTION_AUTHZ = "/vectorstore/collections/docs"
# deliberately nothing to do with the collection convention
PROJECT_AUTHZ = "/programs/foo/projects/bar"
OTHER_AUTHZ = "/open"
UNGRANTED_AUTHZ = "/programs/secret"

COLLECTION = {"collection_name": "docs", "description": "d", "dimensions": 3, "vector_type": "vector"}


@pytest.fixture
def docs_collection(client, allow_authz_paths):
    """Grant the collection plus two unrelated resource paths, and create the collection."""
    allow_authz_paths(COLLECTION_AUTHZ, PROJECT_AUTHZ, OTHER_AUTHZ)
    created = client.post("/vectorstore/collections", json=COLLECTION)
    assert created.status_code == 200, created.text


def _embedding_authz(client, embedding_id: str) -> str:
    """Read an embedding back and return the authz it is stored under."""
    response = client.get(f"/vectorstore/collections/docs/embeddings/{embedding_id}")
    assert response.status_code == 200, response.text
    return response.json()["info"]["authz"]


def test_post_stores_an_authz_outside_the_collection_convention(client, docs_collection):
    """A create can put an embedding under any resource path the caller holds."""
    created = client.post(
        "/vectorstore/collections/docs/embeddings",
        json={"authz": PROJECT_AUTHZ, "embeddings": [{"embedding": [1.0, 0.0, 0.0]}]},
    )
    assert created.status_code == 200, created.text

    embedding_id = created.json()["embeddings"][0]["embedding_id"]
    assert _embedding_authz(client, embedding_id) == PROJECT_AUTHZ


def test_authz_defaults_to_the_collection_path_when_not_supplied(client, docs_collection):
    """The collection path is a default, not a constraint."""
    created = client.post(
        "/vectorstore/collections/docs/embeddings",
        json={"embeddings": [{"embedding": [0.0, 1.0, 0.0]}]},
    )
    assert created.status_code == 200, created.text

    embedding_id = created.json()["embeddings"][0]["embedding_id"]
    assert _embedding_authz(client, embedding_id) == COLLECTION_AUTHZ


def test_put_can_change_an_existing_embeddings_authz(client, docs_collection):
    """An embedding's authz is mutable: PUT moves it from one granted resource to another."""
    created = client.post(
        "/vectorstore/collections/docs/embeddings",
        json={"authz": PROJECT_AUTHZ, "embeddings": [{"embedding": [1.0, 0.0, 0.0]}]},
    )
    assert created.status_code == 200, created.text
    embedding_id = created.json()["embeddings"][0]["embedding_id"]
    assert _embedding_authz(client, embedding_id) == PROJECT_AUTHZ

    updated = client.put(
        f"/vectorstore/collections/docs/embeddings/{embedding_id}",
        json={"authz": OTHER_AUTHZ},
    )
    assert updated.status_code == 200, updated.text
    assert _embedding_authz(client, embedding_id) == OTHER_AUTHZ

    # and back again, to a collection-shaped path this time
    back = client.put(
        f"/vectorstore/collections/docs/embeddings/{embedding_id}",
        json={"authz": COLLECTION_AUTHZ},
    )
    assert back.status_code == 200, back.text
    assert _embedding_authz(client, embedding_id) == COLLECTION_AUTHZ


def test_embeddings_under_arbitrary_authz_are_searchable(client, docs_collection):
    """
    A row whose authz is not collection-shaped is still visible to its grant holder.

    If `allowed_authz` were narrowed the way `allowed_collection_names` is, this row would be
    written and then immediately invisible.
    """
    created = client.post(
        "/vectorstore/collections/docs/embeddings",
        json={"authz": PROJECT_AUTHZ, "embeddings": [{"embedding": [1.0, 0.0, 0.0]}]},
    )
    assert created.status_code == 200, created.text

    search = client.post("/vectorstore/collections/docs/search", json={"input": [1.0, 0.0, 0.0], "top_k": 5})
    assert search.status_code == 200, search.text
    assert len(search.json()["embeddings"]) == 1

    listed = client.get("/vectorstore/collections/docs/embeddings")
    assert listed.status_code == 200, listed.text
    assert len(listed.json()["embeddings"]) == 1


def test_embeddings_are_hidden_when_their_authz_is_not_granted(client, allow_authz_paths, docs_collection):
    """
    Narrowing the grant hides the row, without hiding the collection.

    This is the read half, and it is why a denied read is a 404/empty rather than a 403: the
    policy's USING clause makes the row absent, it does not raise.
    """
    created = client.post(
        "/vectorstore/collections/docs/embeddings",
        json={"authz": PROJECT_AUTHZ, "embeddings": [{"embedding": [1.0, 0.0, 0.0]}]},
    )
    assert created.status_code == 200, created.text
    embedding_id = created.json()["embeddings"][0]["embedding_id"]

    # keep the collection readable, drop the project grant
    allow_authz_paths(COLLECTION_AUTHZ)

    assert client.get("/vectorstore/collections/docs").status_code == 200
    assert client.get(f"/vectorstore/collections/docs/embeddings/{embedding_id}").status_code == 404
    assert client.get("/vectorstore/collections/docs/embeddings").json()["embeddings"] == []


def test_storing_an_ungranted_authz_is_forbidden_not_a_server_error(client, docs_collection):
    """
    A write naming a resource the caller does not hold is a 403.

    In production the policy-engine check rejects this first. Under DEBUG_SKIP_AUTH that
    check is skipped, so the request reaches Postgres and the policy's WITH CHECK rejects it
    -- which is precisely the path that used to surface an asyncpg
    `InsufficientPrivilegeError` as a 500. Reporting a caller's authorization failure as a
    server fault is what `RowLevelSecurityDeniedError` exists to prevent.
    """
    created = client.post(
        "/vectorstore/collections/docs/embeddings",
        json={"authz": UNGRANTED_AUTHZ, "embeddings": [{"embedding": [1.0, 0.0, 0.0]}]},
    )
    assert created.status_code == 403, created.text
    assert "authz" in created.json()["detail"].lower()


def test_moving_an_embedding_to_an_ungranted_authz_is_forbidden(client, docs_collection):
    """The same guard on the update path, where WITH CHECK applies to the new row."""
    created = client.post(
        "/vectorstore/collections/docs/embeddings",
        json={"authz": PROJECT_AUTHZ, "embeddings": [{"embedding": [1.0, 0.0, 0.0]}]},
    )
    assert created.status_code == 200, created.text
    embedding_id = created.json()["embeddings"][0]["embedding_id"]

    moved = client.put(
        f"/vectorstore/collections/docs/embeddings/{embedding_id}",
        json={"authz": UNGRANTED_AUTHZ},
    )
    assert moved.status_code == 403, moved.text

    # the row is untouched
    assert _embedding_authz(client, embedding_id) == PROJECT_AUTHZ
