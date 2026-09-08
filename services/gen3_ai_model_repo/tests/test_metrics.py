"""Tests for model-repo Prometheus API metrics."""

from fastapi.testclient import TestClient

from gen3_ai_model_repo.main import get_app

COUNTER = "gen3_ai_model_repo_api_requests_total"


def test_served_request_is_counted_with_route_template() -> None:
    """A request is counted using its route template rather than its URL value."""
    client = TestClient(get_app())

    response = client.get("/api/models/ns/repo/revisions")
    metrics = client.get("/metrics").text

    assert response.status_code == 401
    assert f"{COUNTER}{{" in metrics
    assert 'path="/api/models/{namespace}/{repo}/revisions"' in metrics


def test_metrics_endpoint_does_not_count_itself() -> None:
    """Scraping metrics does not inflate the API request counter."""
    client = TestClient(get_app())

    before = client.get("/metrics").text
    client.get("/metrics")
    after = client.get("/metrics").text

    def counter_lines(body: str) -> list[str]:
        return sorted(line for line in body.splitlines() if line.startswith(COUNTER))

    assert counter_lines(after) == counter_lines(before)
