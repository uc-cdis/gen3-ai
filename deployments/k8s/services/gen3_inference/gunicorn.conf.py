import logging

import cdislogging
import gunicorn.glogging
from gen3_inference.config import (
    DEBUG,
    ENABLE_OPENTELEMETRY_TRACES,
    OTEL_EXPORTER_OTLP_ENDPOINT,
    VERBOSE_INTERNAL_LOGS,
)
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
)
from prometheus_client import multiprocess


def child_exit(server, worker):
    """
    Required for Prometheus multiprocess setup
    See: https://prometheus.github.io/client_python/multiprocess/
    """
    multiprocess.mark_process_dead(worker.pid)


class CustomLogger(gunicorn.glogging.Logger):
    """
    Initialize root and gunicorn loggers with cdislogging configuration.
    """

    @staticmethod
    def _remove_handlers(logger):
        """
        Use Python's built-in logging module to remove all handlers associated
        with logger (logging.Logger).
        """
        while logger.handlers:
            logger.removeHandler(logger.handlers[0])

    def __init__(self, cfg):
        """
        Apply cdislogging configuration after gunicorn has set up it's loggers.
        """
        super().__init__(cfg)

        self._remove_handlers(logging.getLogger())
        cdislogging.get_logger(None, log_level="debug" if DEBUG else "info")
        internal_log_level = "warning"

        if VERBOSE_INTERNAL_LOGS:
            internal_log_level = "debug"

        for logger_name in [
            "gunicorn",
            "gunicorn.access",
            "uvicorn.error",
            "httpcore",
            "httpx",
            "asyncio",
        ]:
            self._remove_handlers(logging.getLogger(logger_name))
            cdislogging.get_logger(
                logger_name,
                log_level=internal_log_level,
            )


# see https://opentelemetry-python.readthedocs.io/en/latest/examples/fork-process-model/README.html
def post_fork(server, worker):
    if not ENABLE_OPENTELEMETRY_TRACES:
        server.log.info(f"Disabled OpenTelemetry on worker spawned (pid: {worker.pid})")
        return

    server.log.info(f"Setting up OpenTelemetry on worker spawned (pid: {worker.pid})")

    resource = Resource.create(
        attributes={
            "service.name": "gen3_inference",
            # "If workers are not distinguished within attributes, traces and
            # metrics exported from each worker will be indistinguishable. While
            # not necessarily an issue for traces, it is confusing for almost
            # all metric types. A built-in way to identify a worker is by PID
            # but this may lead to high label cardinality. An alternative
            # workaround and additional discussion are available here:
            # https://github.com/benoitc/gunicorn/issues/1352"
            "worker": worker.pid,
        }
    )

    trace.set_tracer_provider(TracerProvider(resource=resource))

    if OTEL_EXPORTER_OTLP_ENDPOINT:
        span_processor = BatchSpanProcessor(
            OTLPSpanExporter(
                endpoint=OTEL_EXPORTER_OTLP_ENDPOINT,
            )
        )
    else:
        span_processor = BatchSpanProcessor(ConsoleSpanExporter())

    trace.get_tracer_provider().add_span_processor(span_processor)


logger_class = CustomLogger

# wsgi_app = "gen3_inference.main:app_instance"

# override with ENV VAR: GUNICORN_BIND
bind = "0.0.0.0:4143"

# override with ENV VAR: GUNICORN_WORKERS
workers = 4

# default was `30` for the 2 below
# can override with ENV VAR: GUNICORN_ARGS="--timeout 120 --graceful_timeout 120"
timeout = 90
graceful_timeout = 90
