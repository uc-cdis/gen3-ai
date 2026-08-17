"""Tests that the service exposes Prometheus metrics for the API requests it serves."""

import pytest
from fastapi.testclient import TestClient

from gen3_embeddings.main import get_app

COUNTER = "gen3_embeddings_api_requests_total"
# Any metered path works here. An unrouted one keeps the test off the database while still
# going through the whole middleware stack.
METERED_PATH = "/vectorstore/not-a-real-route"


@pytest.fixture
def client() -> TestClient:
    """
    Return a client for the service.

    Returns:
        TestClient: A client bound to a fresh app. Deliberately not used as a context
            manager, because entering one runs startup, which requires a database.
    """
    return TestClient(get_app(), follow_redirects=True)


def scrape(client: TestClient) -> str:
    """
    Return the current contents of the metrics endpoint.

    Args:
        client (TestClient): The client to scrape through.

    Returns:
        str: The exposition-format body served at /metrics.
    """
    return client.get("/metrics").text


def series_for(body: str, path: str) -> list[str]:
    """
    Return the counter samples recorded against a request path.

    Args:
        body (str): An exposition-format metrics body.
        path (str): The request path to look for.

    Returns:
        list[str]: Matching sample lines, empty when the path was never counted.
    """
    return [line for line in body.splitlines() if line.startswith(COUNTER) and f'path="{path}"' in line]


def test_served_request_is_counted(client: TestClient) -> None:
    """A request to a metered endpoint shows up on /metrics labelled with how it was served."""
    response = client.get(METERED_PATH)

    samples = series_for(scrape(client), METERED_PATH)

    assert samples, f"no {COUNTER} sample for {METERED_PATH}"
    assert 'method="GET"' in samples[0]
    assert f'status_code="{response.status_code}"' in samples[0]


def test_unauthenticated_request_is_counted_as_an_unknown_user(client: TestClient) -> None:
    """A request without a token is still counted, attributed to an unknown user."""
    client.get(METERED_PATH)

    samples = series_for(scrape(client), METERED_PATH)

    assert samples
    assert 'user_id="Unknown"' in samples[0]


@pytest.mark.parametrize("path", ["/metrics", "/_status", "/_version"])
def test_excluded_endpoint_is_not_counted(client: TestClient, path: str) -> None:
    """Endpoints exempt from metrics never appear as a counted request path."""
    client.get(path)

    assert not series_for(scrape(client), path)
