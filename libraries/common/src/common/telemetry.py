"""
OpenTelemetry tracing setup shared by all services.
"""

import functools
import inspect
import re
from types import FunctionType, ModuleType
from typing import Any, cast

from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as GrpcSpanExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as HttpSpanExporter
from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SpanExporter

from common.config import (
    ENABLE_OPENTELEMETRY_TRACES,
    ENDPOINTS_WITHOUT_METRICS,
    FORCE_DISABLE_CUSTOM_TRACING,
    OTEL_EXPORTER_OTLP_ENDPOINT,
    OTEL_EXPORTER_OTLP_PROTOCOL,
    logging,
)

GRPC_PROTOCOL = "grpc"

# A marker to be set on already `traced` items. Instrumenting the same class or module twice in one
# process - which every `get_app()` in a test suite does - would otherwise nest a second span
# around every call.
_TRACED_MARKER = "_gen3_traced"

# Set by `no_trace` to keep the module and class walks off a function.
_NO_TRACE_MARKER = "_gen3_no_trace"


def configure_tracing(app: FastAPI, service_name: str) -> None:
    """
    Install a tracer provider and instrument the app to emit request spans.

    Does nothing when ENABLE_OPENTELEMETRY_TRACES is off. Spans go to the configured OTLP
    collector, or to the console when OTEL_EXPORTER_OTLP_ENDPOINT is an empty string.

    Args:
        app (FastAPI): The application to instrument.
        service_name (str): Value for the `service.name` resource attribute.
    """
    if not ENABLE_OPENTELEMETRY_TRACES:
        logging.info("ENABLE_OPENTELEMETRY_TRACES is False, skipping OpenTelemetry setup")
        return

    if _tracer_provider_is_set():
        # A provider installed by something else (e.g. the `opentelemetry-instrument`
        # wrapper) wins: set_tracer_provider ignores the second call and only warns.
        logging.info("A tracer provider is already installed, reusing it")
    else:
        exporter = _span_exporter()

        provider = TracerProvider(resource=Resource.create(attributes={"service.name": service_name}))
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

    # Incoming HTTP requests, minus the endpoints that exist to be polled. Liveness and
    # readiness probes hit these every few seconds per replica, and a span each would
    # swamp real traffic in Tempo.
    FastAPIInstrumentor.instrument_app(app, excluded_urls=excluded_url_patterns())

    # Outbound async HTTP calls
    HTTPXClientInstrumentor().instrument()

    # Outbound sync HTTP calls (captures requests made by libs that are not async)
    RequestsInstrumentor().instrument()

    # Database queries
    AsyncPGInstrumentor().instrument()

    # Log correlation: puts otelTraceID/otelSpanID/otelServiceName on every record, which
    # gen3logging's formatters render.
    LoggingInstrumentor().instrument(inject_trace_context=True)


def get_tracer(name: str) -> trace.Tracer:
    """
    Return a tracer for one instrumentation scope.

    Args:
        name (str): The scope, conventionally the calling module's `__name__`.

    Returns:
        trace.Tracer: A tracer. Safe to hold from import time: until a provider is installed
            this is a proxy, and it starts recording once one is.
    """
    return trace.get_tracer(name)


def traced[Function: FunctionType](fn: Function) -> Function:
    """
    Wrap a function so each call emits a span named `<module>.<qualified name>`.

    Qualified name includes anything inside the module the named function is *in* (e.g.
    if it's in a class, then the qualname is "SomeClass.some_function")

    Args:
        fn (Function): A sync or async function. Bound to `FunctionType` rather than a callable,
            because this reads `fn.__module__` and `fn.__qualname__` and hands `fn` to
            `functools.wraps`, none of which an arbitrary callable object carries.

    Returns:
        Function: A wrapped function, or `fn` itself when custom tracing is disabled or `fn` is
            already wrapped.

    Raises:
        TypeError: If `fn` is a generator or async generator function. A span around one of
            those ends when the generator object is created, so it measures nothing, and the
            wrapper also hides the function's generator-ness from callers that introspect it,
            such as FastAPI's dependency injection. The message says what to do instead, which
            is to open a span inside the function with `get_tracer`.
    """
    if inspect.isgeneratorfunction(fn) or inspect.isasyncgenfunction(fn):
        # Raised even when tracing is off, so the mistake cannot hide behind a config value.
        raise TypeError(
            f"cannot trace generator function {fn.__qualname__}: the span would end when the "
            "generator object is created, before any of the body runs, and wrapping also hides "
            "the function's generator-ness from FastAPI's dependency injection. Instead, open "
            "the span inside the function around the work you want measured: "
            "`with get_tracer(__name__).start_as_current_span('name'): ...`. See docs/observability.md."
        )

    if not _custom_tracing_enabled() or getattr(fn, _TRACED_MARKER, False):
        return fn

    tracer = trace.get_tracer(fn.__module__)
    span_name = f"{fn.__module__}.{fn.__qualname__}"

    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            with tracer.start_as_current_span(span_name):
                return await fn(*args, **kwargs)

        wrapper = async_wrapper
    else:

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with tracer.start_as_current_span(span_name):
                return fn(*args, **kwargs)

        wrapper = sync_wrapper

    setattr(wrapper, _TRACED_MARKER, True)
    return cast(Function, wrapper)


def no_trace[Function: FunctionType](fn: Function) -> Function:
    """
    Mark a function for `instrument_module` and `instrument_class` to skip.

    Use this on a function called once per row or per loop iteration inside a module that is
    otherwise worth tracing, where a span per call would cost more than it reports.

    Args:
        fn (Function): The function to leave alone.

    Returns:
        Function: `fn`, unchanged.
    """
    setattr(fn, _NO_TRACE_MARKER, True)
    return fn


def instrument_class(cls: type) -> None:
    """
    Replace the methods a class defines with traced versions, in place.

    Only the class's own attributes are considered, so inherited methods are left to the class
    that defines them, and dunder methods are skipped.

    A method defined with `@staticmethod`, `@classmethod` or `@property` is skipped too: the
    class dict holds a descriptor for those, not the underlying function, so there is nothing
    here to wrap. To trace one, decorate it where it is defined and keep `@traced` innermost,
    directly above the `def`.

    Args:
        cls (type): The class to instrument.
    """
    if not _custom_tracing_enabled():
        return

    for name, attr in list(vars(cls).items()):
        if not name.startswith("__") and _is_traceable(attr):
            setattr(cls, name, traced(attr))


def instrument_module(module: ModuleType) -> None:
    """
    Replace the functions a module defines with traced versions, in place.

    Functions the module merely imported are skipped, so instrumenting one module cannot
    silently instrument another's code, or a third-party library's.

    Only calls that look the function up on the module are traced. A caller that did
    `from x import work` holds the original function and keeps calling it untraced, so a
    module whose consumers import it that way needs `@traced` at each definition instead.

    Args:
        module (ModuleType): The module to instrument.
    """
    if not _custom_tracing_enabled():
        return

    for name, attr in list(vars(module).items()):
        if _is_traceable(attr) and attr.__module__ == module.__name__:
            setattr(module, name, traced(attr))


def excluded_url_patterns() -> str:
    """
    Build the URL exclusion list for the request instrumentation.

    The instrumentation matches these against a whole URL, `scheme://host/path`, using a search
    rather than a full match, so each pattern is anchored at the end. The ASGI path already
    carries any root_path, which a suffix match tolerates.

    Returns:
        str: A comma-separated list of regexes, in the form
            `opentelemetry.util.http.parse_excluded_urls` expects, covering the endpoints that
            are exempt from metrics.
    """
    return ",".join(sorted(re.escape(path) + "$" for path in ENDPOINTS_WITHOUT_METRICS))


def _span_exporter() -> SpanExporter:
    """
    Build the span exporter selected by OTEL_EXPORTER_OTLP_PROTOCOL.

    An empty OTEL_EXPORTER_OTLP_ENDPOINT selects the console exporter, which is how local
    development inspects spans without a collector.

    Returns:
        SpanExporter: A gRPC, HTTP, or console exporter.
    """
    if not OTEL_EXPORTER_OTLP_ENDPOINT:
        return ConsoleSpanExporter()

    if OTEL_EXPORTER_OTLP_PROTOCOL == GRPC_PROTOCOL:
        return GrpcSpanExporter(endpoint=OTEL_EXPORTER_OTLP_ENDPOINT)

    # OTEL_EXPORTER_OTLP_ENDPOINT is a signal-agnostic base per the OTLP spec, and the HTTP
    # exporter only appends `/v1/traces` when it reads that variable itself. An `endpoint`
    # passed to the constructor is used verbatim, so the path has to be added here.
    return HttpSpanExporter(endpoint=f"{OTEL_EXPORTER_OTLP_ENDPOINT.rstrip('/')}/v1/traces")


def _custom_tracing_enabled() -> bool:
    """
    Report whether the per-function span helpers should wrap anything.

    Read through a function rather than captured in a constant so a test can monkeypatch
    either config value on this module.

    Returns:
        bool: True when tracing is on and the kill switch is off.
    """
    return bool(ENABLE_OPENTELEMETRY_TRACES) and not FORCE_DISABLE_CUSTOM_TRACING


def _is_traceable(attr: object) -> bool:
    """Return whether a class or module attribute is a function `traced` can wrap."""
    return (
        inspect.isfunction(attr)
        and not inspect.isgeneratorfunction(attr)
        and not inspect.isasyncgenfunction(attr)
        and not getattr(attr, _NO_TRACE_MARKER, False)
        # This would indicate it's already traced
        and not getattr(attr, _TRACED_MARKER, False)
    )


def _tracer_provider_is_set() -> bool:
    """Return whether a real tracer provider has already replaced the default proxy."""
    return not isinstance(trace.get_tracer_provider(), trace.ProxyTracerProvider)
