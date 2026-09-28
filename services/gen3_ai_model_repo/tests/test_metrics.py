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


def test_startup_metrics_exposes_repository_gauges(monkeypatch) -> None:
    """A metrics-enabled startup publishes all repository gauges with labels."""
    import gen3_ai_model_repo.main as main

    async def no_database_check():
        """Avoid external services while exercising application startup."""

    async def no_storage_check():
        """Avoid external storage while exercising application startup."""

    async def metrics_values():
        """Return deterministic repository metrics for the scrape."""
        return 2, 3, 4096

    monkeypatch.setattr(main, "check_db_connection", no_database_check)
    monkeypatch.setattr(main, "initialize_storage", no_storage_check)
    monkeypatch.setattr(main, "get_repository_metrics", metrics_values)
    with TestClient(get_app()) as client:
        body = client.get("/metrics").text

    assert "gen3_ai_model_repo_models_count{" in body and 'service="gen3_ai_model_repo"} 2.0' in body
    assert "gen3_ai_model_repo_files_count{" in body and 'service="gen3_ai_model_repo"} 3.0' in body
    assert "gen3_ai_model_repo_total_size_bytes{" in body and 'service="gen3_ai_model_repo"} 4096.0' in body
