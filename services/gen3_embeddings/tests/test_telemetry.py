"""Tests that building the service wires up tracing for its own internals."""

from typing import cast

import pytest
from asyncpg import Pool
from fastapi.testclient import TestClient
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from common import telemetry
from gen3_embeddings.database.db import DataAccessLayer
from gen3_embeddings.main import get_app

TRACED_ROUTE = "/vectorstore/collections/a-specific-collection"


@pytest.fixture(scope="session")
def exporter() -> InMemorySpanExporter:
    """
    Return the exporter collecting spans, installed as the process's tracer provider.

    Returns:
        InMemorySpanExporter: Finished spans, in the order they ended.
    """
    # Installed before any app is built so configure_tracing finds a provider already in place
    # and reuses it, instead of exporting to a collector that is not running. Session scoped
    # because only the first provider installed in a process takes effect.
    in_memory = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(in_memory))
    trace.set_tracer_provider(provider)
    return in_memory


@pytest.fixture
def spans(exporter: InMemorySpanExporter) -> InMemorySpanExporter:
    """
    Return an exporter holding only the spans produced after this fixture ran.

    Returns:
        InMemorySpanExporter: The session exporter, emptied.
    """
    exporter.clear()
    return exporter


@pytest.fixture
def traced_app(monkeypatch: pytest.MonkeyPatch, exporter: InMemorySpanExporter) -> TestClient:
    """
    Return a client for an app built with tracing on.

    The suite disables tracing globally in conftest, so the switch is flipped here for the
    duration of one test.

    Returns:
        TestClient: A client for the traced app. Not entered as a context manager, so the
            lifespan's database checks do not run.
    """
    monkeypatch.setattr(telemetry, "ENABLE_OPENTELEMETRY_TRACES", True)
    return TestClient(get_app())


def names(spans: InMemorySpanExporter) -> list[str]:
    """
    Return the names of the spans recorded so far.

    Args:
        spans (InMemorySpanExporter): The exporter to read.

    Returns:
        list[str]: Span names, in the order the spans ended.
    """
    return [span.name for span in spans.get_finished_spans()]


def test_served_request_produces_a_span(traced_app: TestClient, spans: InMemorySpanExporter) -> None:
    """A request through the traced app records a span for the request itself."""
    traced_app.patch(TRACED_ROUTE)

    assert names(spans)


def test_data_access_layer_calls_are_traced(traced_app: TestClient, spans: InMemorySpanExporter) -> None:
    """Building the app leaves the data access layer emitting a span per call."""
    # The one data access call reachable without a database: it reads allowed_authz and
    # never touches the pool.
    layer = DataAccessLayer(pool=cast(Pool, None), allowed_authz=[TRACED_ROUTE])
    layer._get_allowed_collection_names_from_allowed_authz()

    assert "gen3_embeddings.database.db.DataAccessLayer._get_allowed_collection_names_from_allowed_authz" in names(
        spans
    )


def test_kill_switch_leaves_request_tracing_alone(
    monkeypatch: pytest.MonkeyPatch, exporter: InMemorySpanExporter
) -> None:
    """FORCE_DISABLE_CUSTOM_TRACING drops the function spans only."""
    monkeypatch.setattr(telemetry, "ENABLE_OPENTELEMETRY_TRACES", True)
    monkeypatch.setattr(telemetry, "FORCE_DISABLE_CUSTOM_TRACING", True)
    client = TestClient(get_app())
    exporter.clear()

    client.patch(TRACED_ROUTE)

    assert names(exporter)
