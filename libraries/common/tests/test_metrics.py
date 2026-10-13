"""Tests metrics"""

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from common import config
from common.metrics import get_metrics_client


def test_disabled_metrics_still_answer_scrapes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    With metrics off, the app still builds and /metrics answers 200 with no samples.

    The multiprocess directory is only created when metrics are on, so it is pointed somewhere
    that does not exist, as it would be in a fresh container. A 404 instead of an empty 200
    would make Prometheus report the target as down.
    """
    monkeypatch.setattr(config, "ENABLE_METRICS", False)
    monkeypatch.setattr(config, "PROMETHEUS_MULTIPROC_DIR", str(tmp_path / "never-created"))
    app = FastAPI()

    metrics_client = get_metrics_client(app)
    response = TestClient(app, follow_redirects=True).get("/metrics")

    assert not metrics_client.enabled
    assert response.status_code == 200
    assert response.text == ""
