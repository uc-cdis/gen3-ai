"""
Tests for the request limits in `gen3_embeddings.config`, `.limits`, and the schemas.

These limits exist to stop one request from consuming unbounded memory or CPU, so the
assertions are about a request being *refused* rather than about what it would have
returned. Each test names the resource the limit protects.

The first half needs no database: those limits are enforced before any handler runs, either
by the ASGI middleware or by Pydantic. The second half goes through the real app, to confirm
the bound is actually wired into the route rather than only defined.
"""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from gen3_embeddings.config import (
    MAX_AI_MODEL_NAME_LENGTH,
    MAX_COLLECTION_NAME_LENGTH,
    MAX_COLLECTIONS_PER_SEARCH,
    MAX_COLLECTIONS_QUERY_LENGTH,
    MAX_EMBEDDING_UUIDS_PER_REQUEST,
    MAX_EMBEDDINGS_PER_REQUEST,
    MAX_METADATA_BYTES,
    MAX_METADATA_DEPTH,
    MAX_METADATA_KEYS,
    MAX_PAGE,
    MAX_PAGE_SIZE,
    MAX_SEARCH_FILTER_VALUE_LENGTH,
    MAX_SEARCH_FILTERS,
    MAX_TEXT_INPUT_LENGTH,
    MAX_TOP_K,
    MAX_VECTOR_DIMENSIONS,
)
from gen3_embeddings.limits import RequestSizeLimitMiddleware, metadata_depth, validate_metadata
from gen3_embeddings.models.helpers import normalize_collection_name
from gen3_embeddings.models.schemas import (
    CreateCollectionBody,
    CreateEmbeddingsBody,
    SearchRequestBody,
    UpdateEmbeddingBody,
)

# A vector width every test collection uses, small enough to keep request bodies readable.
DIMENSIONS = 3


def make_collection(client, name: str = "docs", dimensions: int = DIMENSIONS) -> None:
    """
    Create a collection to exercise a limit against.

    Args:
        client: The test client.
        name (str): Collection name to create.
        dimensions (int): Vector width for the collection.
    """
    resp = client.post(
        "/vectorstore/collections",
        json={
            "collection_name": name,
            "description": name,
            "dimensions": dimensions,
            "vector_type": "vector",
        },
    )
    assert resp.status_code == 200, resp.text


class TestRequestSizeLimitMiddleware:
    """
    The byte ceiling on the request body.

    This is the only limit that can act before the body is read, so it is the one that
    protects the JSON parse itself. Tested against a bare app rather than the service, since
    nothing about it is service-specific.
    """

    @staticmethod
    def build_client(max_body_bytes: int) -> TestClient:
        """
        Build a client for a minimal app wrapped in the middleware.

        Args:
            max_body_bytes (int): Limit to configure the middleware with.

        Returns:
            TestClient: Client for an app whose only route echoes the body length.
        """
        app = FastAPI()

        @app.post("/echo")
        async def echo(payload: dict) -> dict:
            return {"keys": len(payload)}

        app.add_middleware(RequestSizeLimitMiddleware, max_body_bytes=max_body_bytes)
        return TestClient(app)

    def test_body_within_limit_is_passed_through_intact(self):
        """A body under the limit reaches the handler unchanged."""
        client = self.build_client(max_body_bytes=1024)
        resp = client.post("/echo", json={"a": 1, "b": 2})
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"keys": 2}

    def test_declared_content_length_over_limit_is_refused(self):
        """A Content-Length over the limit is refused without the body being parsed."""
        client = self.build_client(max_body_bytes=64)
        resp = client.post("/echo", content=b'{"a": "' + b"x" * 512 + b'"}')
        assert resp.status_code == 413, resp.text
        assert "too large" in resp.json()["detail"]

    def test_unannounced_body_over_limit_is_refused(self):
        """A streamed body with no Content-Length is counted as it arrives, then refused."""
        client = self.build_client(max_body_bytes=64)

        def chunks():
            for _ in range(10):
                yield b"x" * 32

        resp = client.post("/echo", content=chunks(), headers={"content-type": "application/json"})
        assert resp.status_code == 413, resp.text

    def test_unannounced_body_within_limit_is_replayed_to_the_app(self):
        """A streamed body under the limit is buffered and handed on without being lost."""
        client = self.build_client(max_body_bytes=1024)
        body = json.dumps({"a": 1, "b": 2, "c": 3}).encode()

        def chunks():
            yield body[:5]
            yield body[5:]

        resp = client.post("/echo", content=chunks(), headers={"content-type": "application/json"})
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"keys": 3}

    def test_unparseable_content_length_is_a_client_error(self):
        """A Content-Length we cannot compare against the limit is refused, not ignored."""
        client = self.build_client(max_body_bytes=64)
        resp = client.post(
            "/echo",
            content=b"{}",
            headers={"content-length": "not-a-number", "content-type": "application/json"},
        )
        assert resp.status_code == 400, resp.text


class TestMetadataDepth:
    """The nesting measurement behind the metadata depth limit."""

    def test_scalar_has_no_depth(self):
        """A value that cannot nest measures zero."""
        assert metadata_depth("x") == 0
        assert metadata_depth(1) == 0
        assert metadata_depth({}) == 0

    def test_depth_counts_each_container_level(self):
        """Each nested dict or list counts as one level, whichever the container is."""
        assert metadata_depth({"a": 1}) == 1
        assert metadata_depth({"a": {"b": 1}}) == 2
        assert metadata_depth({"a": [{"b": 1}]}) == 3

    def test_measurement_stops_just_past_the_limit(self):
        """
        A pathologically nested document costs no more to measure than a compliant one.

        The point of the cap is that this call must not walk 100_000 levels to find out that
        the document is too deep, so the returned depth is only ever one past the limit.
        """
        deep = {}
        for _ in range(100_000):
            deep = {"a": deep}
        assert metadata_depth(deep) == MAX_METADATA_DEPTH + 1


class TestValidateMetadata:
    """The three metadata limits, which Pydantic cannot express as field constraints."""

    def test_none_and_compliant_metadata_pass_through_unchanged(self):
        """Validation is a gate, not a transform: what goes in comes back out."""
        assert validate_metadata(None) is None
        metadata = {"source": "file.md", "nested": {"a": 1}}
        assert validate_metadata(metadata) == metadata

    def test_too_many_top_level_keys_is_refused(self):
        """Key count is bounded because every key is stored and returned on each read."""
        metadata = {f"k{i}": 1 for i in range(MAX_METADATA_KEYS + 1)}
        with pytest.raises(ValueError, match="top-level keys"):
            validate_metadata(metadata)

    def test_too_deep_is_refused(self):
        """Depth is bounded separately from size, because nesting costs stack, not bytes."""
        deep = {"leaf": 1}
        for _ in range(MAX_METADATA_DEPTH + 1):
            deep = {"a": deep}
        with pytest.raises(ValueError, match="nest at most"):
            validate_metadata(deep)

    def test_too_large_when_serialized_is_refused(self):
        """Serialized size is the bound that matters, since that is what gets stored."""
        metadata = {"blob": "x" * (MAX_METADATA_BYTES + 1)}
        with pytest.raises(ValueError, match="bytes when serialized"):
            validate_metadata(metadata)

    def test_depth_is_checked_before_size(self):
        """
        Depth has to be checked first, or the size check crashes on a deep document.

        `json.dumps` recurses, so serializing something nested past Python's recursion limit
        raises RecursionError rather than the ValueError a caller should see.
        """
        deep = {"leaf": 1}
        for _ in range(5000):
            deep = {"a": deep}
        with pytest.raises(ValueError, match="nest at most"):
            validate_metadata(deep)


class TestSchemaFieldLimits:
    """
    Bounds declared on the request schemas.

    Checked against the models directly, so a failure points at the schema rather than at
    whichever route happened to use it.
    """

    def test_vector_longer_than_the_maximum_is_refused(self):
        """Vector width is bounded independently of any collection's `dimensions`."""
        with pytest.raises(ValidationError, match="too_long"):
            UpdateEmbeddingBody(embedding=[0.0] * (MAX_VECTOR_DIMENSIONS + 1))

    def test_non_finite_vector_components_are_refused(self):
        """pgvector rejects NaN and infinity, so they are refused at the edge instead."""
        for bad in (float("nan"), float("inf"), float("-inf")):
            with pytest.raises(ValidationError):
                UpdateEmbeddingBody(embedding=[bad, 0.0, 0.0])

    def test_more_embeddings_than_the_per_request_maximum_is_refused(self):
        """Item count multiplies against the per-vector bound, so it is bounded too."""
        with pytest.raises(ValidationError, match="too_long"):
            CreateEmbeddingsBody(embeddings=[{"embedding": [0.0]}] * (MAX_EMBEDDINGS_PER_REQUEST + 1))

    def test_oversized_metadata_on_a_created_embedding_is_refused(self):
        """The metadata limits apply on create, not only on update."""
        with pytest.raises(ValidationError, match="bytes when serialized"):
            CreateEmbeddingsBody(
                embeddings=[{"embedding": [0.0], "metadata": {"blob": "x" * (MAX_METADATA_BYTES + 1)}}]
            )

    def test_dimensions_over_the_maximum_is_refused_at_collection_creation(self):
        """
        The `dimensions` bound is the one that matters most.

        It is checked once here, but every embedding later written to the collection is
        validated against it, so it sets the ceiling for that collection's rows forever.
        """
        with pytest.raises(ValidationError):
            CreateCollectionBody(collection_name="docs", dimensions=MAX_VECTOR_DIMENSIONS + 1)

    def test_top_k_over_the_maximum_is_refused(self):
        """`top_k` becomes a SQL LIMIT over rows that each carry a full vector."""
        with pytest.raises(ValidationError):
            SearchRequestBody(input=[0.0], top_k=MAX_TOP_K + 1)

    def test_too_many_search_filters_is_refused(self):
        """Each filter adds a WHERE clause and two parameters to a generated statement."""
        filters = {f"k{i}": "v" for i in range(MAX_SEARCH_FILTERS + 1)}
        with pytest.raises(ValidationError, match="too_long"):
            SearchRequestBody(input=[0.0], filters=filters)

    def test_oversized_search_filter_value_is_refused(self):
        """A filter value is compared in SQL, so its length is bounded like any other field."""
        with pytest.raises(ValidationError, match="too_long"):
            SearchRequestBody(input=[0.0], filters={"k": "v" * (MAX_SEARCH_FILTER_VALUE_LENGTH + 1)})

    def test_raw_text_search_input_is_bounded(self):
        """
        Raw text search is not implemented, but the field still has to be bounded.

        An unbounded string costs the memory to parse it whether or not the handler
        goes on to use it.
        """
        with pytest.raises(ValidationError):
            SearchRequestBody(input="x" * (MAX_TEXT_INPUT_LENGTH + 1))


class TestCollectionNameLimits:
    """The bound on a collection name, which is the only bound a path segment gets."""

    def test_name_longer_than_the_maximum_is_refused(self):
        """A path parameter has no schema to constrain it, so the normalizer does it."""
        with pytest.raises(ValueError, match="at most"):
            normalize_collection_name("a" * (MAX_COLLECTION_NAME_LENGTH + 1))

    def test_name_at_the_maximum_is_accepted(self):
        """The bound is inclusive, so the longest permitted name still works."""
        name = "a" * MAX_COLLECTION_NAME_LENGTH
        assert normalize_collection_name(name) == name

    def test_oversized_name_in_a_path_is_a_client_error(self, client, allow_authz):
        """Through the app, an over-long name is a 400 rather than a database round trip."""
        allow_authz("docs")
        resp = client.get(f"/vectorstore/collections/{'a' * (MAX_COLLECTION_NAME_LENGTH + 1)}")
        assert resp.status_code == 400, resp.text
        assert "at most" in resp.json()["detail"]


class TestRouteLimits:
    """
    The same limits seen through the real app.

    Defining a bound and wiring it into the route are two different things, so these go
    through HTTP to confirm the second.
    """

    def test_oversized_vector_on_create_is_refused(self, client, allow_authz):
        """Refused on width before the dimension mismatch check, which would also reject it."""
        allow_authz("docs")
        make_collection(client)

        resp = client.post(
            "/vectorstore/collections/docs/embeddings",
            json={"embeddings": [{"embedding": [0.0] * (MAX_VECTOR_DIMENSIONS + 1)}]},
        )
        assert resp.status_code == 422, resp.text

    def test_too_many_embeddings_on_create_is_refused(self, client, allow_authz):
        """One request cannot ask for an unbounded number of row writes."""
        allow_authz("docs")
        make_collection(client)

        resp = client.post(
            "/vectorstore/collections/docs/embeddings",
            json={"embeddings": [{"embedding": [0.0] * DIMENSIONS}] * (MAX_EMBEDDINGS_PER_REQUEST + 1)},
        )
        assert resp.status_code == 422, resp.text

    def test_oversized_metadata_on_create_is_refused(self, client, allow_authz):
        """Metadata is stored as jsonb and returned on every read, so its size is bounded."""
        allow_authz("docs")
        make_collection(client)

        resp = client.post(
            "/vectorstore/collections/docs/embeddings",
            json={
                "embeddings": [
                    {
                        "embedding": [0.0] * DIMENSIONS,
                        "metadata": {"blob": "x" * (MAX_METADATA_BYTES + 1)},
                    }
                ]
            },
        )
        assert resp.status_code == 422, resp.text

    def test_top_k_over_the_maximum_is_refused(self, client, allow_authz):
        """Bounds how many full vectors a single search can be made to return."""
        allow_authz("docs")
        make_collection(client)

        resp = client.post(
            "/vectorstore/collections/docs/search",
            json={"input": [1.0, 0.0, 0.0], "top_k": MAX_TOP_K + 1},
        )
        assert resp.status_code == 422, resp.text

    def test_page_beyond_the_maximum_is_refused(self, client, allow_authz):
        """Bounds the OFFSET handed to Postgres, and the depth of scan we will serve."""
        allow_authz("docs")
        make_collection(client)

        for path in ("/vectorstore/collections", "/vectorstore/collections/docs/embeddings"):
            resp = client.get(path, params={"page": MAX_PAGE + 1})
            assert resp.status_code == 422, f"{path}: {resp.text}"

    def test_page_size_is_bounded_above_but_not_below(self, client, allow_authz):
        """
        Both list endpoints cap `page_size` and both accept a small one.

        The cap is the limit that matters, since it bounds how many full vectors one
        response carries.
        """
        allow_authz("docs")
        make_collection(client)

        for path in ("/vectorstore/collections", "/vectorstore/collections/docs/embeddings"):
            too_big = client.get(path, params={"page_size": MAX_PAGE_SIZE + 1})
            assert too_big.status_code == 422, f"{path}: {too_big.text}"

            smallest = client.get(path, params={"page_size": 1})
            assert smallest.status_code == 200, f"{path}: {smallest.text}"
            assert smallest.json()["page_size"] == 1

    def test_too_many_bulk_uuids_is_refused(self, client, allow_authz):
        """The request is small, but each UUID that resolves returns a full vector."""
        allow_authz("docs")
        make_collection(client)

        uuids = [f"00000000-0000-0000-0000-{i:012d}" for i in range(MAX_EMBEDDING_UUIDS_PER_REQUEST + 1)]

        for path in ("/embeddings/bulk", "/vectorstore/collections/docs/embeddings/bulk"):
            resp = client.post(path, json=uuids)
            assert resp.status_code == 422, f"{path}: {resp.text}"

    def test_empty_bulk_uuid_list_is_refused(self, client, allow_authz):
        """A bulk read with nothing to read is a client mistake, not an empty result."""
        allow_authz("docs")
        make_collection(client)

        resp = client.post("/embeddings/bulk", json=[])
        assert resp.status_code == 422, resp.text

    def test_oversized_ai_model_name_is_refused(self, client, allow_authz):
        """Not wired up to anything yet, but still caller-controlled free text."""
        allow_authz("docs")
        make_collection(client)

        resp = client.post(
            "/vectorstore/collections/docs/search",
            params={"ai_model": "m" * (MAX_AI_MODEL_NAME_LENGTH + 1)},
            json={"input": [1.0, 0.0, 0.0]},
        )
        assert resp.status_code == 422, resp.text

    def test_too_many_collections_in_one_search_is_refused(self, client, allow_authz):
        """Each named collection costs its own database round trip."""
        allow_authz("docs")
        make_collection(client)

        names = ",".join(f"c{i}" for i in range(MAX_COLLECTIONS_PER_SEARCH + 1))
        resp = client.post(
            "/vectorstore/search",
            params={"collections": names},
            json={"input": [1.0, 0.0, 0.0]},
        )
        assert resp.status_code == 400, resp.text
        assert "at once" in resp.json()["detail"]

    def test_search_over_more_authorized_collections_than_one_search_may_span_is_refused(
        self, client, allow_authz, monkeypatch
    ):
        """
        Searching every authorized collection refuses past the bound instead of truncating.

        This is the branch with no `collections` parameter to bound it, so the ceiling has
        to be applied to what the caller is authorized for. Truncating would be worse than
        refusing: `list_collections` orders by name, so a caller over the bound would get
        the alphabetically-first collections searched and a ranking that looks complete.

        The bound is lowered rather than creating its default's worth of collections. The
        route reads it as a module global, so that is where it has to be patched.
        """
        from gen3_embeddings.routes import search as search_module

        monkeypatch.setattr(search_module, "MAX_COLLECTIONS_SEARCHED", 2)

        allow_authz("c_a", "c_b", "c_c")
        for name in ("c_a", "c_b", "c_c"):
            make_collection(client, name)

        resp = client.post("/vectorstore/search", json={"input": [1.0, 0.0, 0.0]})
        assert resp.status_code == 400, resp.text
        detail = resp.json()["detail"]
        assert "more than 2 collections" in detail
        # The refusal has to say how to get the results anyway, since it is refusing the
        # only request that needs no prior knowledge of which collections exist.
        assert "collections" in detail and "top_k" in detail

    def test_search_at_exactly_the_collection_ceiling_is_served(self, client, allow_authz, monkeypatch):
        """
        The bound is a ceiling, not a threshold: a caller right at it is still served.

        Worth pinning separately, because the route detects the over-limit case by asking
        for one collection more than the ceiling, and an off-by-one there would refuse the
        callers it is meant to allow.
        """
        from gen3_embeddings.routes import search as search_module

        monkeypatch.setattr(search_module, "MAX_COLLECTIONS_SEARCHED", 3)

        allow_authz("c_a", "c_b", "c_c")
        for name in ("c_a", "c_b", "c_c"):
            make_collection(client, name)

        resp = client.post("/vectorstore/search", json={"input": [1.0, 0.0, 0.0]})
        assert resp.status_code == 200, resp.text

    def test_oversized_collections_query_string_is_refused_before_splitting(self, client, allow_authz):
        """
        The string is bounded before the split, because the split is what allocates.

        A value with no commas in it would pass the count check, so the length bound is the
        only thing standing between a megabyte-long parameter and a megabyte of list.
        """
        allow_authz("docs")
        make_collection(client)

        resp = client.post(
            "/vectorstore/search",
            params={"collections": "a" * (MAX_COLLECTIONS_QUERY_LENGTH + 1)},
            json={"input": [1.0, 0.0, 0.0]},
        )
        assert resp.status_code == 422, resp.text

    def test_request_body_over_the_byte_limit_is_refused(self, monkeypatch):
        """
        The real app refuses an over-large body rather than parsing it.

        The configured default is far too large to build in a test, so the limit is lowered
        and the app rebuilt. No database is needed and no lifespan is started: the whole
        point of this limit is that it answers before the application runs at all.
        """
        from gen3_embeddings import config as config_module
        from gen3_embeddings import main as main_module
        from gen3_embeddings.main import get_app

        monkeypatch.setattr(config_module, "MAX_REQUEST_BODY_BYTES", 128)
        monkeypatch.setattr(main_module.config, "MAX_REQUEST_BODY_BYTES", 128)

        client = TestClient(get_app())
        resp = client.post(
            "/vectorstore/collections",
            content=json.dumps({"collection_name": "docs", "dimensions": 3, "pad": "x" * 512}),
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 413, resp.text
        assert "too large" in resp.json()["detail"]
