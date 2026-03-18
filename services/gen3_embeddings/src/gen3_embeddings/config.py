import os

import cdislogging
from starlette.config import Config
from starlette.datastructures import Secret

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

CONFIG_PATH = os.path.abspath(os.getenv("CONFIG_PATH", f"{CURRENT_DIR}/../../.env"))

config = Config(CONFIG_PATH)
DEBUG = config("DEBUG", cast=bool, default=False)

# this turns on debug logging for certain noisy internal libraries
# Note: the list of libraries is in the gunicorn.conf.py
VERBOSE_INTERNAL_LOGS = config("VERBOSE_INTERNAL_LOGS", cast=bool, default=False)

logging = cdislogging.get_logger(__name__, log_level="debug" if DEBUG else "info")

# will skip authorization when a token is not provided. note that if a token is provided, then
# auth will still occur
DEBUG_SKIP_AUTH = config("DEBUG_SKIP_AUTH", cast=bool, default=False)

logging.info(f"DEBUG is {DEBUG}")
logging.info(f"VERBOSE_INTERNAL_LOGS is {VERBOSE_INTERNAL_LOGS}")

if DEBUG_SKIP_AUTH:
    logging.warning(
        f"DEBUG_SKIP_AUTH is {DEBUG_SKIP_AUTH}. Authorization will be SKIPPED if no token is provided. "
        "FOR NON-PRODUCTION USE ONLY!! USE WITH CAUTION!!"
    )

DB_DRIVER = config("DB_DRIVER", default="postgresql")
DB_USER = config("DB_USER", default="postgres")
DB_PASSWORD = config("DB_PASSWORD", cast=Secret, default=None)
DB_HOST = config("DB_HOST", default="localhost")
DB_PORT = config("DB_PORT", cast=int, default="5432")
DB_DATABASE = config("DB_DATABASE", default="testgen3embeddings")

DB_CONNECTION_STRING = config(
    "DB_CONNECTION_STRING",
    cast=Secret,
    default=f"{DB_DRIVER}://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_DATABASE}",
)

URL_PREFIX = config("GEN3_EMBEDDINGS_PROXY_URL_PREFIX", default="", cast=str)

# enable Prometheus Metrics for observability purposes
#
# WARNING: Any counters, gauges, histograms, etc. should be carefully
# reviewed to make sure its labels do not contain any PII / PHI.
#
# IMPORTANT: This enables a /metrics endpoint which is OPEN TO ALL TRAFFIC, unless controlled upstream
ENABLE_PROMETHEUS_METRICS = config("ENABLE_PROMETHEUS_METRICS", cast=bool, default=False)

PROMETHEUS_MULTIPROC_DIR = config("PROMETHEUS_MULTIPROC_DIR", default="/var/tmp/prometheus_metrics", cast=str)

# Location of the policy engine service, Arborist
# Defaults to the default service name in k8s magic DNS setup
ARBORIST_URL = config("ARBORIST_URL", default="http://arborist-service", cast=str)

PUBLIC_ROUTES = {"/", "/_status", "/_status/", "/_version", "/_version/"}
ENDPOINTS_WITHOUT_METRICS = {"/metrics", "/metrics/"} | PUBLIC_ROUTES

# If you're using a trusted upstream reverse proxy and want to preserve
# the x-forwarded-for chain in calls to the underlying EMBEDDINGS server,
# set IS_UPSTREAM_CALLER_TRUSTED_REVERSE_PROXY to true
IS_UPSTREAM_CALLER_TRUSTED_REVERSE_PROXY = config("IS_UPSTREAM_CALLER_TRUSTED_REVERSE_PROXY", cast=bool, default=False)

# WARNING: Careful changing these, they require close sync with the authorization source
#          of truth. This is the "service" passed to Gen3 Authz for authorization checks
#          and the Authz resource corresponding to the use of the service itself.
#          Additional authorization is applied on a per-EMBEDDINGS Resource level within
#          this proxy service, these are a first gate for API-level access. See the
#          rest of the docs/service for more info on EMBEDDINGS authz.
AUTHZ_SERVICE_NAME = config("GEN3_EMBEDDINGS_PROXY_AUTHZ_SERVICE_NAME", default="gen3-embeddings", cast=str)
AUTHZ_SERVICE_RESOURCE = config(
    "GEN3_EMBEDDINGS_PROXY_AUTHZ_SERVICE_NAME",
    default="/services/gen3-embeddings",
    cast=str,
)

# This app exports traces using OpenTelemetry. By default in Gen3, we use Alloy for collection.
ENABLE_OPENTELEMETRY_TRACES = config("ENABLE_OPENTELEMETRY_TRACES", cast=bool, default=True)
# For local development, set this to an EMPTY STRING and it will output to console. See gunicorn.conf.py
OTEL_EXPORTER_OTLP_ENDPOINT = config("OTEL_EXPORTER_OTLP_ENDPOINT", default="http://alloy.monitoring.4318", cast=str)

ASYNC_HTTP_CLIENT_TIMEOUT = config("ASYNC_HTTP_CLIENT_TIMEOUT", cast=float, default=30)
