"""
Tests for where a collection name is canonicalized, and where it deliberately is not.

The rule this pins down:

- Request parameters (path and create body) are normalized at the HTTP boundary, so the
  authorization check and the database lookup compare the same string.
- Authz resource paths from the policy engine are used verbatim. Arborist resource paths are
  case-sensitive, so this service must not rewrite what a grant says; a policy has to name
  the collection in its stored (normalized) form.
"""

import pytest

from gen3_embeddings.auth import get_allowed_collection_names_from_authz


@pytest.mark.parametrize(
    "authz_paths, expected",
    [
        (["/vectorstore/collections/docs"], {"docs"}),
        (["/vectorstore/collections/docs", "/vectorstore/collections/images"], {"docs", "images"}),
        # NOT normalized: Arborist is case-sensitive, so a grant is taken exactly as written
        (["/vectorstore/collections/Docs"], {"Docs"}),
        # the bare base resource does not currently grant every collection
        (["/vectorstore/collections"], set()),
        # deeper paths are not collection grants
        (["/vectorstore/collections/docs/embeddings"], set()),
        # unrelated resources are ignored
        (["/programs/foo"], set()),
    ],
)
def test_authz_paths_are_used_verbatim(authz_paths, expected):
    """Grants are read exactly as the policy engine reports them."""
    assert get_allowed_collection_names_from_authz(authz_paths) == expected


def test_grant_must_name_the_collection_as_stored(client, allow_authz):
    """
    A grant whose casing differs from the stored name does not authorize access.

    This is intentional rather than a bug: Arborist resource paths are case-sensitive, so
    the operator must write the policy against the collection's normalized name. Pinned as a
    test so the behavior is a decision rather than an accident.
    """
    allow_authz("Docs")

    response = client.post(
        "/vectorstore/collections",
        json={"collection_name": "Docs", "description": "d", "dimensions": 3, "vector_type": "vector"},
    )
    assert response.status_code == 403
    assert "Not authorized" in response.json()["detail"]


@pytest.mark.parametrize("requested", ["docs", "DOCS", "Docs", "  docs  "])
def test_authz_check_uses_the_normalized_name(client, allow_authz, monkeypatch, requested):
    """
    The Arborist resource path is built from the normalized name.

    This is what the `CollectionName` annotation uniquely buys: the authorization check runs
    as a route dependency, before any DAL call, so if the path parameter were still raw then
    Arborist would be asked about `/vectorstore/collections/DOCS` while the DAL compared
    against `docs`. Since Arborist is case-sensitive, those two gates would disagree.

    Tests bypass Arborist via DEBUG_SKIP_AUTH, so assert on the resource path that would be
    sent rather than on the response status.
    """
    from gen3_embeddings import auth as auth_module

    allow_authz("docs")
    create = client.post(
        "/vectorstore/collections",
        json={"collection_name": "docs", "description": "d", "dimensions": 3, "vector_type": "vector"},
    )
    assert create.status_code == 200, create.text

    checked_resources: list[list[str]] = []

    async def fake_authorize_request(*, request, authz_access_method, authz_resources):
        checked_resources.append(authz_resources)

    monkeypatch.setattr(auth_module, "authorize_request", fake_authorize_request)

    response = client.get(f"/vectorstore/collections/{requested}")
    assert response.status_code == 200, response.text

    assert checked_resources, "the authz dependency did not run"
    assert checked_resources[0] == ["/vectorstore/collections/docs"]


@pytest.mark.parametrize("requested", ["docs", "DOCS", "Docs"])
def test_path_parameter_is_normalized_before_lookup(client, allow_authz, requested):
    """
    Any casing in the URL reaches the same collection, since the stored name is canonical.

    Note this also holds via the DAL's own normalization in `get_collection_by_name`, so it
    documents the end-to-end behavior rather than isolating the boundary annotation. See
    `test_authz_check_uses_the_normalized_name` for the part only the annotation provides.
    """
    allow_authz("docs")
    create = client.post(
        "/vectorstore/collections",
        json={"collection_name": "docs", "description": "d", "dimensions": 3, "vector_type": "vector"},
    )
    assert create.status_code == 200, create.text

    response = client.get(f"/vectorstore/collections/{requested}")
    assert response.status_code == 200, response.text
    assert response.json()["collection_name"] == "docs"


def test_create_body_is_normalized(client, allow_authz):
    """The create body is canonicalized too, so the stored name is always normalized."""
    allow_authz("docs")

    response = client.post(
        "/vectorstore/collections",
        json={"collection_name": "DOCS", "description": "d", "dimensions": 3, "vector_type": "vector"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["collection_name"] == "docs"


@pytest.mark.parametrize(
    "path",
    [
        "/vectorstore/collections/bad.name",
        "/vectorstore/collections/bad.name/embeddings",
    ],
)
def test_invalid_collection_name_in_path_is_a_400(client, allow_authz, path):
    """An unusable name is rejected at the boundary with the documented 400, not a 422."""
    allow_authz("docs")

    response = client.get(path)
    assert response.status_code == 400, response.text
    assert "collection_name may only contain" in response.json()["detail"]


def test_invalid_collection_name_in_body_is_a_400(client, allow_authz):
    """The create body is validated the same way as a path parameter."""
    allow_authz("docs")

    response = client.post(
        "/vectorstore/collections",
        json={"collection_name": "bad.name", "description": "d", "dimensions": 3, "vector_type": "vector"},
    )
    assert response.status_code == 400, response.text
    assert "collection_name may only contain" in response.json()["detail"]


@pytest.mark.parametrize("requested", ["DOCS", "Docs"])
def test_update_with_mixed_case_path_still_finds_the_collection(client, allow_authz, requested):
    """
    PATCH resolves the collection through the normalized path parameter.

    Unlike reads, `dal.update_collection` does not normalize internally, so the boundary
    annotation is the only thing making this work.
    """
    allow_authz("docs")
    create = client.post(
        "/vectorstore/collections",
        json={"collection_name": "docs", "description": "before", "dimensions": 3, "vector_type": "vector"},
    )
    assert create.status_code == 200, create.text

    response = client.patch(f"/vectorstore/collections/{requested}", json={"description": "after"})
    assert response.status_code == 200, response.text
    assert response.json()["description"] == "after"


@pytest.mark.parametrize("requested", ["DOCS", "Docs"])
def test_delete_with_mixed_case_path_still_finds_the_collection(client, allow_authz, requested):
    """DELETE likewise relies on the boundary annotation; the DAL does not normalize."""
    allow_authz("docs")
    create = client.post(
        "/vectorstore/collections",
        json={"collection_name": "docs", "description": "d", "dimensions": 3, "vector_type": "vector"},
    )
    assert create.status_code == 200, create.text

    assert client.delete(f"/vectorstore/collections/{requested}").status_code == 204
    assert client.get("/vectorstore/collections/docs").status_code == 404
