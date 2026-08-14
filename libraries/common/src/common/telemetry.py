"""
OpenTelemetry tracing setup shared by all services.
"""

from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as GrpcSpanExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as HttpSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SpanExporter

from common.config import (
    ENABLE_OPENTELEMETRY_TRACES,
    OTEL_EXPORTER_OTLP_ENDPOINT,
    OTEL_EXPORTER_OTLP_PROTOCOL,
    logging,
)

GRPC_PROTOCOL = "grpc"


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

    FastAPIInstrumentor.instrument_app(app)


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


def _tracer_provider_is_set() -> bool:
    """Report whether a real tracer provider has already replaced the default proxy."""
    return not isinstance(trace.get_tracer_provider(), trace.ProxyTracerProvider)
