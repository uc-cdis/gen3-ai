"""Tests that the service exposes Prometheus metrics for the API requests it serves."""

import pytest
from fastapi.testclient import TestClient

from gen3_embeddings.main import get_app

COUNTER = "gen3_embeddings_api_requests_total"

TEMPLATED_ROUTE = "/vectorstore/collections/{collection_name}"
COLLECTION_NAME = "a-specific-collection"


@pytest.fixture
def client() -> TestClient:
    """
    Return a client for the service.

    Returns:
        TestClient: A client bound to a fresh app. Deliberately not used as a context
            manager, because entering one runs startup, which requires a database.
    """
    return TestClient(get_app(), follow_redirects=True)


def samples(client: TestClient, path: str) -> list[str]:
    """
    Return the counter samples currently recorded against a path label.

    Args:
        client (TestClient): The client to scrape /metrics through.
        path (str): The path label to look for.

    Returns:
        list[str]: Matching sample lines, empty when that path was never counted.
    """
    body = client.get("/metrics").text
    return [line for line in body.splitlines() if line.startswith(COUNTER) and f'path="{path}"' in line]


def counter_lines(client: TestClient) -> list[str]:
    """
    Return every counter sample currently exposed, sorted.

    Args:
        client (TestClient): The client to scrape /metrics through.

    Returns:
        list[str]: Sample lines for the API request counter.
    """
    body = client.get("/metrics").text
    return sorted(line for line in body.splitlines() if line.startswith(COUNTER))


def test_served_request_is_counted(client: TestClient) -> None:
    """A request to a metered endpoint shows up on /metrics labelled with how it was served."""
    response = client.patch(f"/vectorstore/collections/{COLLECTION_NAME}")

    recorded = samples(client, TEMPLATED_ROUTE)

    assert recorded, f"no {COUNTER} sample for {TEMPLATED_ROUTE}"
    assert 'method="PATCH"' in recorded[0]
    assert f'status_code="{response.status_code}"' in recorded[0]


def test_path_parameter_is_not_recorded_as_a_label(client: TestClient) -> None:
    """
    A path parameter is recorded as its route template, not its value.

    Recording the value would create a new time series per collection.
    """
    client.patch(f"/vectorstore/collections/{COLLECTION_NAME}")

    assert samples(client, TEMPLATED_ROUTE)
    assert not samples(client, f"/vectorstore/collections/{COLLECTION_NAME}")


def test_unrouted_request_is_counted_under_one_shared_label(client: TestClient) -> None:
    """Requests matching no route share a single path label rather than one per URL."""
    client.get("/vectorstore/no-such-route")
    client.get("/also-not-a-route")

    assert samples(client, "<unmatched>")
    assert not samples(client, "/vectorstore/no-such-route")


def test_unauthenticated_request_is_counted_as_an_unknown_user(client: TestClient) -> None:
    """A request without a token is still counted, attributed to an unknown user."""
    client.patch(f"/vectorstore/collections/{COLLECTION_NAME}")

    recorded = samples(client, TEMPLATED_ROUTE)

    assert recorded
    assert 'user_id="Unknown"' in recorded[0]


@pytest.mark.parametrize("path", ["/metrics", "/_status", "/_version"])
def test_excluded_endpoint_is_not_counted(client: TestClient, path: str) -> None:
    """Endpoints exempt from metrics never appear as a counted request path."""
    client.get(path)

    assert not samples(client, path)


def test_site_root_is_counted(client: TestClient) -> None:
    """The site root is public but still metered."""
    client.get("/", follow_redirects=False)

    assert samples(client, "/")


@pytest.mark.parametrize("path", ["/docs", "/openapi.json"])
def test_browsable_documentation_is_counted(client: TestClient, path: str) -> None:
    """Docs and spec traffic is metered, so it is possible to see who reads the API."""
    client.get(path)

    assert samples(client, path)


def test_scraping_metrics_never_counts_itself(client: TestClient) -> None:
    """
    Scraping the metrics endpoint records nothing, so a scrape cannot inflate its own counts.

    Compares every counter line before and after, rather than looking for a `/metrics` label:
    the endpoint is mounted rather than routed, so a regression here surfaces under whatever
    fallback label the middleware lands on.
    """
    before = counter_lines(client)

    client.get("/metrics")
    client.get("/metrics")

    assert counter_lines(client) == before
