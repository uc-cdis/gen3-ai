"""Tests for the per-function span helpers in common.telemetry."""

import asyncio
import inspect
from types import ModuleType

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from opentelemetry.util.http import parse_excluded_urls

from common import telemetry
from common.telemetry import excluded_url_patterns, instrument_class, instrument_module, no_trace, traced

SLEEP_SECONDS = 0.05


@pytest.fixture(scope="session")
def exporter() -> InMemorySpanExporter:
    """
    Return the exporter collecting every span the suite produces.

    Returns:
        InMemorySpanExporter: Finished spans, in the order they ended.
    """
    # Session scoped because set_tracer_provider only warns on a second call: the first
    # provider installed in a process is the one every tracer resolves to.
    in_memory = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(in_memory))
    trace.set_tracer_provider(provider)
    return in_memory


@pytest.fixture(autouse=True)
def spans(exporter: InMemorySpanExporter) -> InMemorySpanExporter:
    """
    Return an exporter holding only the spans of the test that asks for it.

    Returns:
        InMemorySpanExporter: The session exporter, emptied.
    """
    exporter.clear()
    return exporter


@pytest.fixture(autouse=True)
def tracing_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enable the helpers regardless of the environment the suite runs in."""
    monkeypatch.setattr(telemetry, "ENABLE_OPENTELEMETRY_TRACES", True)
    monkeypatch.setattr(telemetry, "FORCE_DISABLE_CUSTOM_TRACING", False)


def names(spans: InMemorySpanExporter) -> list[str]:
    """
    Return the names of the spans recorded so far.

    Args:
        spans (InMemorySpanExporter): The exporter to read.

    Returns:
        list[str]: Span names, in the order the spans ended.
    """
    return [span.name for span in spans.get_finished_spans()]


def duration_ns(span: ReadableSpan) -> int:
    """
    Return how long a finished span lasted, in nanoseconds.

    Args:
        span (ReadableSpan): A span that has ended.

    Returns:
        int: The elapsed time.

    Raises:
        AssertionError: If the span never started or never ended, which the SDK represents as
            an absent timestamp.
    """
    assert span.start_time is not None
    assert span.end_time is not None
    return span.end_time - span.start_time


def build_module(source: str, name: str = "sample_module") -> ModuleType:
    """
    Return a module built from source, for the walkers to instrument.

    Args:
        source (str): Module body. Functions it defines report `name` as their `__module__`.
        name (str): The module's name.

    Returns:
        ModuleType: The populated module.
    """
    module = ModuleType(name)
    exec(source, module.__dict__)  # noqa: S102
    return module


def test_sync_function_is_traced_under_its_module_and_qualname(spans: InMemorySpanExporter) -> None:
    """A traced sync function returns its value and records one span named for itself."""

    @traced
    def add(a: int, b: int) -> int:
        return a + b

    assert add(1, 2) == 3
    assert names(spans) == [
        f"{__name__}.{test_sync_function_is_traced_under_its_module_and_qualname.__qualname__}.<locals>.add"
    ]


def test_async_function_is_traced(spans: InMemorySpanExporter) -> None:
    """A traced async function returns its value and records one span."""

    @traced
    async def add(a: int, b: int) -> int:
        return a + b

    assert asyncio.run(add(1, 2)) == 3
    assert len(names(spans)) == 1


def test_async_span_covers_work_after_an_await(spans: InMemorySpanExporter) -> None:
    """An async function's span measures the whole call, not just coroutine creation."""

    @traced
    async def slow() -> None:
        await asyncio.sleep(SLEEP_SECONDS)

    asyncio.run(slow())

    assert duration_ns(spans.get_finished_spans()[0]) >= SLEEP_SECONDS * 1e9


def test_nested_calls_nest_spans(spans: InMemorySpanExporter) -> None:
    """A traced function called from another becomes a child span, not a sibling."""

    @traced
    def inner() -> None:
        pass

    @traced
    def outer() -> None:
        inner()

    outer()

    inner_span, outer_span = spans.get_finished_spans()
    assert inner_span.parent is not None
    assert inner_span.parent.span_id == outer_span.context.span_id


def test_exception_is_recorded_and_still_propagates(spans: InMemorySpanExporter) -> None:
    """A failing traced function reports an ERROR span and lets the exception through."""

    @traced
    def boom() -> None:
        raise ValueError("uh oh")

    with pytest.raises(ValueError):
        boom()

    span = spans.get_finished_spans()[0]
    assert span.status.status_code is StatusCode.ERROR
    assert [event.name for event in span.events] == ["exception"]


def test_generator_function_cannot_be_traced() -> None:
    """Tracing a generator function is refused rather than silently producing an empty span."""

    def numbers():
        yield 1

    with pytest.raises(TypeError):
        traced(numbers)


def test_async_generator_function_cannot_be_traced() -> None:
    """Tracing an async generator is refused, since it would also break FastAPI's dep injection."""

    async def numbers():
        yield 1

    with pytest.raises(TypeError):
        traced(numbers)


def test_generator_function_is_refused_even_when_tracing_is_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """The generator refusal does not depend on configuration, so it cannot hide in prod."""
    monkeypatch.setattr(telemetry, "FORCE_DISABLE_CUSTOM_TRACING", True)

    def numbers():
        yield 1

    with pytest.raises(TypeError):
        traced(numbers)


def test_class_methods_are_traced(spans: InMemorySpanExporter) -> None:
    """Instrumenting a class traces the methods it defines, including private ones."""

    class Service:
        def public(self) -> None:
            self._private()

        def _private(self) -> None:
            pass

    instrument_class(Service)
    Service().public()

    assert [name.split(".")[-1] for name in names(spans)] == ["_private", "public"]


def test_instrumenting_a_class_twice_produces_one_span_per_call(spans: InMemorySpanExporter) -> None:
    """A second walk over the same class does not wrap the wrapper."""

    class Service:
        def work(self) -> None:
            pass

    instrument_class(Service)
    instrument_class(Service)
    Service().work()

    assert len(names(spans)) == 1


def test_class_instrumentation_skips_dunders(spans: InMemorySpanExporter) -> None:
    """Constructing an instrumented class records no span for __init__."""

    class Service:
        def __init__(self) -> None:
            pass

    instrument_class(Service)
    Service()

    assert names(spans) == []


def test_module_functions_are_traced(spans: InMemorySpanExporter) -> None:
    """Instrumenting a module traces the functions it defines."""
    module = build_module("def work():\n    return 1\n")

    instrument_module(module)

    assert module.work() == 1
    assert names(spans) == ["sample_module.work"]


def test_module_instrumentation_skips_imported_functions(spans: InMemorySpanExporter) -> None:
    """A function a module imported belongs to its own module and is left alone."""
    module = build_module("from common.telemetry import get_tracer\n")

    instrument_module(module)

    module.get_tracer("x")
    assert names(spans) == []


def test_module_instrumentation_leaves_async_generators_intact(spans: InMemorySpanExporter) -> None:
    """An async generator survives a walk unwrapped, since wrapping it would break FastAPI dep injection."""
    module = build_module("async def dependency():\n    yield 1\n")

    instrument_module(module)

    assert inspect.isasyncgenfunction(module.dependency)


def test_module_instrumentation_leaves_generators_intact() -> None:
    """A sync generator survives a walk unwrapped."""
    module = build_module("def numbers():\n    yield 1\n")

    instrument_module(module)

    assert inspect.isgeneratorfunction(module.numbers)


def test_no_trace_excludes_a_function_from_a_walk(spans: InMemorySpanExporter) -> None:
    """A function marked with no_trace is skipped while its neighbours are traced."""
    module = build_module("def hot():\n    return 1\n\n\ndef cold():\n    return 2\n")
    no_trace(module.hot)

    instrument_module(module)

    module.hot()
    module.cold()
    assert names(spans) == ["sample_module.cold"]


@pytest.mark.parametrize(
    ("flag", "value"),
    [("ENABLE_OPENTELEMETRY_TRACES", False), ("FORCE_DISABLE_CUSTOM_TRACING", True)],
)
def test_disabled_tracing_wraps_nothing(
    spans: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch, flag: str, value: bool
) -> None:
    """Either switch off leaves functions, classes and modules completely unwrapped."""
    monkeypatch.setattr(telemetry, flag, value)

    def work() -> int:
        return 1

    class Service:
        def work(self) -> None:
            pass

    module = build_module("def work():\n    return 1\n")

    assert traced(work) is work
    instrument_class(Service)
    instrument_module(module)

    Service().work()
    module.work()
    assert names(spans) == []


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
    assert parse_excluded_urls(excluded_url_patterns()).url_disabled(url)


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
    assert not parse_excluded_urls(excluded_url_patterns()).url_disabled(url)
