"""
Common configuration which is used by all services.
"""

import os
import sys
from pathlib import Path

import gen3logging
from starlette.config import Config


def get_venv_root() -> Path | None:
    """
    Return the root of the current virtual environment.

    Returns:
        Path | None: The absolute path to the environment's root, or None when the interpreter
            is running from the system Python.
    """
    if hasattr(sys, "base_prefix"):
        if sys.prefix != sys.base_prefix:
            return Path(sys.prefix).parent

    return None


# NOTE: Default config only works when:
#       The .env is in its standard location:
#       /services/{service_name}/.env
#       AND the common library is installed in a virtualenv for the service
#       AND the virtualenv directory is in:
#       /services/{service_name}/{venv_name}
CURRENT_DIR = get_venv_root() or os.path.dirname(os.path.realpath(__file__))
CONFIG_PATH = os.path.abspath(os.getenv("CONFIG_PATH", f"{CURRENT_DIR}/.env"))

# Containers configure everything through environment variables and ship no .env, and
# starlette emits a UserWarning for a missing env_file, so only pass one that exists.
CONFIG_FILE_EXISTS = os.path.isfile(CONFIG_PATH)

starlette_config = Config(CONFIG_PATH if CONFIG_FILE_EXISTS else None)
DEBUG = starlette_config("DEBUG", cast=bool, default=False)

# this turns on debug logging for certain noisy internal libraries
# Note: the list of libraries is in common/logging_setup.py
VERBOSE_INTERNAL_LOGS = starlette_config("VERBOSE_INTERNAL_LOGS", cast=bool, default=False)

# Emit one JSON object per log line instead of the human-readable text format. Named to match the
# env var gen3logging reads on its own; passing it explicitly means this default wins over the
# library's.
GEN3_JSON_LOGS = starlette_config("GEN3_JSON_LOGS", cast=bool, default=True)

logging = gen3logging.get_logger(__name__, log_level="debug" if DEBUG else "info", json_logs=GEN3_JSON_LOGS)

if CONFIG_FILE_EXISTS:
    logging.info(f"Using configuration file: {CONFIG_PATH}")
else:
    logging.info(f"No configuration file at {CONFIG_PATH}, using environment variables only")

# will skip authorization when a token is not provided. note that if a token is provided, then
# auth will still occur
DEBUG_SKIP_AUTH = starlette_config("DEBUG_SKIP_AUTH", cast=bool, default=False)

# this will effectively turn off authorization checking,
# allowing for anyone to use the AI functionality
ALLOW_ANONYMOUS_ACCESS = starlette_config("ALLOW_ANONYMOUS_ACCESS", cast=bool, default=False)

logging.info(f"DEBUG is {DEBUG}")
logging.info(f"VERBOSE_INTERNAL_LOGS is {VERBOSE_INTERNAL_LOGS}")
logging.info(f"GEN3_JSON_LOGS is {GEN3_JSON_LOGS}")

if DEBUG_SKIP_AUTH:
    logging.warning(
        f"DEBUG_SKIP_AUTH is {DEBUG_SKIP_AUTH}. Authorization will be SKIPPED if no token is provided. "
        "FOR NON-PRODUCTION USE ONLY!! USE WITH CAUTION!!"
    )
if ALLOW_ANONYMOUS_ACCESS:
    logging.warning(
        f"ALLOW_ANONYMOUS_ACCESS is {ALLOW_ANONYMOUS_ACCESS}. Authorization will be SKIPPED. "
        "ENSURE THIS IS ACCEPTABLE!!"
    )

# Location of the policy engine service, Arborist
# Defaults to the default service name in k8s magic DNS setup
ARBORIST_URL = starlette_config("ARBORIST_URL", default="http://arborist-service", cast=str)

# Location of these AI services
GEN3_AI_MODEL_REPO_URL = starlette_config(
    "GEN3_AI_MODEL_REPO_URL", default="http://gen3-ai-model-repo-service", cast=str
)
GEN3_EMBEDDINGS_URL = starlette_config("GEN3_EMBEDDINGS_URL", default="http://gen3-embeddings-service", cast=str)
GEN3_INFERENCE_URL = starlette_config("GEN3_INFERENCE_URL", default="http://gen3-inference-service", cast=str)

# Endpoints that exist to be polled or fetched by a browser, and are not worth observability
# (metrics or tracing).
#
# Do NOT add "/" here. common/telemetry.py turns each of these into a regex anchored at the end
# of the URL, so a bare "/" becomes "/$", which matches every URL ending in a slash and would
# silently drop the trailing-slash form of every real route from tracing. The site root, the
# docs and the OpenAPI spec are all left out deliberately: traffic to them is cheap to record
# and worth seeing.
UNMONITORED_ROUTES = {
    "/_status",
    "/_status/",
    "/_version",
    "/_version/",
    "/favicon.ico",
    "/favicon.ico/",
}
ENDPOINTS_WITHOUT_METRICS = {"/metrics", "/metrics/"} | UNMONITORED_ROUTES

# This app exports traces using OpenTelemetry. By default in Gen3, we use Alloy for collection.
ENABLE_OPENTELEMETRY_TRACES = starlette_config("ENABLE_OPENTELEMETRY_TRACES", cast=bool, default=True)
# For local development, set this to an EMPTY STRING and it will output to console. See common/telemetry.py
OTEL_EXPORTER_OTLP_ENDPOINT = starlette_config(
    "OTEL_EXPORTER_OTLP_ENDPOINT", default="http://alloy.monitoring:4318", cast=str
)
# Alloy accepts both. `http/protobuf` pairs with port 4318 and `grpc` with 4317; a mismatched
# pair fails at export time, not at startup.
OTEL_EXPORTER_OTLP_PROTOCOL = starlette_config("OTEL_EXPORTER_OTLP_PROTOCOL", default="http/protobuf", cast=str)
# Kill switch for the per-function spans that common/telemetry.py's `traced` and `instrument_*`
# helpers add. Turning this on leaves the library instrumentation (FastAPI, asyncpg, httpx,
# requests, logging) running, so requests and queries are still traced with less detail.
FORCE_DISABLE_CUSTOM_TRACING = starlette_config("FORCE_DISABLE_CUSTOM_TRACING", cast=bool, default=False)

# `traced` decides whether to wrap when the module it decorates is imported, which is before any
# app factory runs, so cdispyutils reads these two from the environment rather than from anything
# passed to `configure_tracing`. Publishing what CONFIG_PATH resolved is what makes a `.env` file
# reach the per-function spans; without it, turning tracing off in that file would still leave
# every `@traced` function wrapped.
os.environ["ENABLE_OPENTELEMETRY_TRACES"] = str(ENABLE_OPENTELEMETRY_TRACES).lower()
os.environ["FORCE_DISABLE_CUSTOM_TRACING"] = str(FORCE_DISABLE_CUSTOM_TRACING).lower()

# Continuous profiling with Pyroscope. See common/profiling.py.
#
# Off by default, unlike tracing: the Pyroscope SDK has no console exporter to fall back on, so
# with nothing listening it retries pushes for the life of the process.
ENABLE_CONTINUOUS_PROFILING = starlette_config("ENABLE_CONTINUOUS_PROFILING", cast=bool, default=False)
# The SDK pushes to `<address>/push.v1.PusherService/Push`, which is Pyroscope's own ingest API
# and not OTLP, so this cannot point at the OTLP ports (4317/4318) that traces use. In Gen3 it is
# Alloy's `pyroscope.receive_http` listener; a Pyroscope server or Grafana Cloud works too.
PYROSCOPE_SERVER_ADDRESS = starlette_config(
    "PYROSCOPE_SERVER_ADDRESS", default="http://alloy.monitoring:4040", cast=str
)
# Samples per second per thread.
PYROSCOPE_SAMPLE_RATE = starlette_config("PYROSCOPE_SAMPLE_RATE", cast=int, default=100)
# Seconds between pushes to the server.
PYROSCOPE_UPLOAD_INTERVAL = starlette_config("PYROSCOPE_UPLOAD_INTERVAL", cast=int, default=10)
PROFILE_CPU = starlette_config("PROFILE_CPU", cast=bool, default=True)
# Allocation profiling. Off by default because it samples the allocator itself, which costs more
# than the CPU profiler on an endpoint that allocates per row. Turn it on to chase a leak.
PROFILE_MEMORY = starlette_config("PROFILE_MEMORY", cast=bool, default=False)
# True measures CPU time, False measures wall clock. These services are I/O bound, so the CPU-only
# default answers "what is burning CPU" and shows nothing for the time a request spends awaiting
# the database. Flip it to attribute latency rather than CPU, and expect the flamegraph to be
# dominated by the event loop waiting.
PROFILE_ON_CPU_ONLY = starlette_config("PROFILE_ON_CPU_ONLY", cast=bool, default=True)
# Credentials for a Pyroscope that requires them, e.g. Grafana Cloud. Empty means unauthenticated,
# which is what an in-cluster Alloy or Pyroscope expects.
PYROSCOPE_BASIC_AUTH_USERNAME = starlette_config("PYROSCOPE_BASIC_AUTH_USERNAME", default="", cast=str)
PYROSCOPE_BASIC_AUTH_PASSWORD = starlette_config("PYROSCOPE_BASIC_AUTH_PASSWORD", default="", cast=str)
PYROSCOPE_TENANT_ID = starlette_config("PYROSCOPE_TENANT_ID", default="", cast=str)

ASYNC_HTTP_CLIENT_TIMEOUT = starlette_config("ASYNC_HTTP_CLIENT_TIMEOUT", cast=float, default=30)

# Metrics provider, at the moment we only support "prometheus". If you want to use a different one,
# you will need to implement the common interface in common/metrics.py.
ENABLE_METRICS = starlette_config("ENABLE_METRICS", default=True, cast=bool)
METRICS_PROVIDER = starlette_config("METRICS_PROVIDER", default="prometheus", cast=str)
PROMETHEUS_MULTIPROC_DIR = starlette_config("PROMETHEUS_MULTIPROC_DIR", default="/var/tmp/prometheus_metrics", cast=str)
# prometheus_client decides between its in-memory and its multiprocess value class once, when it
# is first imported, based on this env var. Anything that sets the var later - including
# cdispyutils' BaseMetrics, which sets it in its constructor - leaves counters in memory while
# /metrics serves a multiprocess registry reading an empty directory, i.e. a 200 with no data.
os.environ.setdefault("PROMETHEUS_MULTIPROC_DIR", PROMETHEUS_MULTIPROC_DIR)
