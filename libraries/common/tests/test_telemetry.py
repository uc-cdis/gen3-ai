"""
Tests the tracing setup common/telemetry.py hands to every service.

The span helpers themselves - `traced`, `no_trace`, `instrument_class`, `instrument_module` -
live in `gen3pyutils.observability.tracing` and are tested there. What is checked here is the
set of paths common.config exempts, and that outbound HTTP from both client libraries in use is
traced.
"""

import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import httpx
import httpx2
import pytest
from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
from opentelemetry.instrumentation.httpx import HTTPX2ClientInstrumentor, HTTPXClientInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util.http import parse_excluded_urls

from common import config, telemetry
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


@pytest.fixture(scope="module")
def upstream() -> Iterator[str]:
    """
    Serve an empty 200 for any GET, standing in for a service this one calls.

    Yields:
        str: The base URL to request.
    """

    class EmptyHandler(BaseHTTPRequestHandler):
        """Answers every GET with an empty body."""

        def do_GET(self) -> None:  # noqa: N802 - name is fixed by BaseHTTPRequestHandler
            """Return 200 with no content."""
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - overridden signature
            """Keep request lines out of the test output."""

    server = HTTPServer(("127.0.0.1", 0), EmptyHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


@pytest.fixture
def spans(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemorySpanExporter]:
    """
    Configure tracing the way a service does, collecting spans in memory.

    `configure_tracing` patches the HTTP client libraries for the whole process, so the patches
    are removed afterwards rather than left to change how later tests' requests behave.

    Yields:
        InMemorySpanExporter: Spans finished while the test ran.
    """
    # Only the first provider installed in a process takes effect, so attach to it if one is
    # already there. `configure_tracing` then reuses it instead of exporting to a collector.
    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(config, "ENABLE_OPENTELEMETRY_TRACES", True)

    telemetry.configure_tracing(FastAPI(), "test-service")
    yield exporter

    for instrumentor in (
        HTTPXClientInstrumentor(),
        HTTPX2ClientInstrumentor(),
        RequestsInstrumentor(),
        AsyncPGInstrumentor(),
        # what gen3pyutils' LoggingInstrumentorWithContext wraps; the wrapper cannot uninstrument
        LoggingInstrumentor(),
    ):
        instrumentor.uninstrument()
    telemetry.reset_tracing_state()
    exporter.shutdown()


@pytest.mark.parametrize(
    "client_module",
    [
        # gen3authz's calls to the policy engine
        httpx,
        # authutils' OIDC discovery and JWKS fetches
        httpx2,
    ],
    ids=lambda module: module.__name__,
)
def test_outbound_http_is_traced(client_module: Any, upstream: str, spans: InMemorySpanExporter) -> None:
    """A request made with either HTTP client library records a client span for its URL."""
    url = f"{upstream}/jwt/keys"

    with client_module.Client() as client:
        client.get(url).raise_for_status()

    client_spans = [
        dict(span.attributes or {}) for span in spans.get_finished_spans() if span.kind == trace.SpanKind.CLIENT
    ]
    # The attribute name depends on which HTTP semantic conventions the instrumentor emits.
    assert [attributes.get("url.full") or attributes.get("http.url") for attributes in client_spans] == [url]
