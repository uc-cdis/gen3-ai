"""
Tests that the endpoints this monorepo exempts from observability are the ones excluded.

The span helpers themselves - `traced`, `no_trace`, `instrument_class`, `instrument_module` -
live in `cdispyutils.observability.tracing` and are tested there. What is checked here is the
set of paths common.config hands them.
"""

import pytest
from opentelemetry.util.http import parse_excluded_urls

from common.config import ENDPOINTS_WITHOUT_METRICS
from common.telemetry import excluded_url_patterns


@pytest.mark.parametrize(
    "url",
    [
        "http://svc/_status",
        "http://svc/_status/",
        "http://svc/metrics",
        "http://svc/ai/embeddings/_status",
    ],
)
def test_polled_endpoints_are_not_traced(url: str) -> None:
    """A probe endpoint is excluded, including under a URL prefix."""
    assert parse_excluded_urls(excluded_url_patterns(ENDPOINTS_WITHOUT_METRICS)).url_disabled(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://svc/",
        "http://svc/docs",
        "http://svc/openapi.json",
        "http://svc/vectorstore/collections",
        "http://svc/vectorstore/collections/",
        "http://svc/vectorstore/collections/a-collection",
    ],
)
def test_real_traffic_is_traced(url: str) -> None:
    """Real routes stay traced, including the trailing-slash form and the site root."""
    assert not parse_excluded_urls(excluded_url_patterns(ENDPOINTS_WITHOUT_METRICS)).url_disabled(url)
