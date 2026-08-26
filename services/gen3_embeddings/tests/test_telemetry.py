"""Tests that building the service wires up tracing for its own internals."""

from collections.abc import Iterator
from typing import cast

import pytest
from asyncpg import Pool
from cdispyutils.observability import continuous_profiling
from fastapi.testclient import TestClient
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from common import config, telemetry
from gen3_embeddings.database.db import DataAccessLayer
from gen3_embeddings.main import get_app

TRACED_ROUTE = "/vectorstore/collections/a-specific-collection"
# Set by the Pyroscope span processor, and what Grafana reads to offer a profile for a span.
PROFILE_ID_ATTRIBUTE = "pyroscope.profile.id"


class StubSDK:
    """Stands in for the Pyroscope SDK, accepting every call `configure_profiling` makes."""

    def configure(self, **kwargs: object) -> None:
        """Accept the agent settings without starting an agent."""

    def shutdown(self) -> None:
        """Accept the request to stop."""


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
    monkeypatch.setattr(config, "ENABLE_OPENTELEMETRY_TRACES", True)
    return TestClient(get_app())


@pytest.fixture(autouse=True)
def stop_profiling_after_each_test(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """
    Leave the profiling module thinking no agent is running, whatever the test did.

    Requests monkeypatch so this finalizer runs while the stubbed SDK is still installed, rather
    than reaching the real one.

    Yields:
        None: Control to the test.
    """
    yield

    continuous_profiling.stop_profiling()
    # `configure_tracing` records what it resolved in a process-wide override, which the
    # instrument helpers read. Left set, the first test to enable tracing decides for the rest.
    telemetry.reset_tracing_state()


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


def test_spans_link_to_profiles_only_when_profiling_is_active(
    monkeypatch: pytest.MonkeyPatch, exporter: InMemorySpanExporter
) -> None:
    """A span carries a profile id when the agent is running, and none when it is not."""
    # Both halves live in one test because the span processor cannot be removed from a tracer
    # provider once added, and the provider is process-wide. Split in two, whichever test ran
    # second would see the other's processor still tagging spans.
    monkeypatch.setattr(config, "ENABLE_OPENTELEMETRY_TRACES", True)

    unprofiled = TestClient(get_app())
    exporter.clear()
    unprofiled.patch(TRACED_ROUTE)

    assert all(PROFILE_ID_ATTRIBUTE not in (span.attributes or {}) for span in exporter.get_finished_spans())

    # Only the SDK is stubbed, so the app starts the agent by the path production takes. A real
    # agent would spend the rest of the suite retrying pushes at a server that is not there.
    monkeypatch.setattr(continuous_profiling, "pyroscope", StubSDK())
    monkeypatch.setattr(config, "ENABLE_CONTINUOUS_PROFILING", True)

    profiled = TestClient(get_app())
    exporter.clear()
    profiled.patch(TRACED_ROUTE)

    assert any(PROFILE_ID_ATTRIBUTE in (span.attributes or {}) for span in exporter.get_finished_spans())


def test_kill_switch_leaves_request_tracing_alone(
    monkeypatch: pytest.MonkeyPatch, exporter: InMemorySpanExporter
) -> None:
    """FORCE_DISABLE_CUSTOM_TRACING drops the function spans only."""
    monkeypatch.setattr(config, "ENABLE_OPENTELEMETRY_TRACES", True)
    # Read from the environment rather than passed to `configure_tracing`, because `traced` runs
    # at import time, before any call could have told it otherwise.
    monkeypatch.setenv("FORCE_DISABLE_CUSTOM_TRACING", "true")
    client = TestClient(get_app())
    exporter.clear()

    client.patch(TRACED_ROUTE)

    assert names(exporter)
