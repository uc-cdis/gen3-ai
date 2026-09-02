"""
OpenTelemetry tracing shared by all services.

The tracing itself lives in `cdispyutils.observability.tracing`, which is Gen3-wide and
configured through keyword arguments. This module is where those arguments come from
`common.config`, so a service calls `configure_tracing(app, name)` and gets the settings the
rest of the monorepo reads.
"""

from cdispyutils.observability.tracing import LoggingInstrumentorWithContext
from cdispyutils.observability.tracing import configure_tracing as _configure_tracing

# Re-exported so service code imports the span helpers from here and never names the library
# directly. Repeating the name after `as` is what marks an import as a deliberate re-export:
# nothing in this module calls these, so ruff would otherwise prune them as unused.
from cdispyutils.observability.tracing import (
    excluded_url_patterns as excluded_url_patterns,
)
from cdispyutils.observability.tracing import get_tracer as get_tracer
from cdispyutils.observability.tracing import instrument_class as instrument_class
from cdispyutils.observability.tracing import instrument_module as instrument_module
from cdispyutils.observability.tracing import no_trace as no_trace
from cdispyutils.observability.tracing import reset_tracing_state as reset_tracing_state
from cdispyutils.observability.tracing import traced as traced
from fastapi import FastAPI
from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor

from common import config


def configure_tracing(app: FastAPI, service_name: str) -> None:
    """
    Install a tracer provider and instrument the app to emit request spans.

    Does nothing when ENABLE_OPENTELEMETRY_TRACES is off. Spans go to the configured OTLP
    collector, or to the console when OTEL_EXPORTER_OTLP_ENDPOINT is an empty string.

    When the Pyroscope agent is already running, spans are also tagged so Grafana can jump from
    a span to the profile for that request. Call `common.profiling.configure_profiling` first,
    otherwise this cannot know the agent exists and the link is left out.

    Args:
        app (FastAPI): The application to instrument.
        service_name (str): Value for the `service.name` resource attribute.
    """
    _configure_tracing(
        app,
        service_name,
        enabled=config.ENABLE_OPENTELEMETRY_TRACES,
        otlp_endpoint=config.OTEL_EXPORTER_OTLP_ENDPOINT,
        otlp_protocol=config.OTEL_EXPORTER_OTLP_PROTOCOL,
        excluded_urls=config.ENDPOINTS_WITHOUT_METRICS,
        instrumentors=[
            # Outbound async HTTP calls
            HTTPXClientInstrumentor(),
            # Outbound sync HTTP calls (captures requests made by libs that are not async)
            RequestsInstrumentor(),
            # Database queries. Not in the library's default set, because the right database
            # instrumentation differs per service; every service here is on asyncpg.
            AsyncPGInstrumentor(),
            # Log correlation: puts otelTraceID/otelSpanID/otelServiceName on every record, which
            # gen3logging's formatters render.
            LoggingInstrumentorWithContext(),
        ],
    )
