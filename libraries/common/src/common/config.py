import os
import sys
from pathlib import Path

import cdislogging
from starlette.config import Config


def get_venv_root() -> Path | None:
    """
    Return the absolute Path to the root of the current virtual environment,
    or None if the interpreter is running from the system Python.
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

starlette_config = Config(CONFIG_PATH)
DEBUG = starlette_config("DEBUG", cast=bool, default=False)

# this turns on debug logging for certain noisy internal libraries
# Note: the list of libraries is in the gunicorn.conf.py
VERBOSE_INTERNAL_LOGS = starlette_config("VERBOSE_INTERNAL_LOGS", cast=bool, default=False)

logging = cdislogging.get_logger(__name__, log_level="debug" if DEBUG else "info")

logging.info(f"Using configuration file: {CONFIG_PATH}")

# will skip authorization when a token is not provided. note that if a token is provided, then
# auth will still occur
DEBUG_SKIP_AUTH = starlette_config("DEBUG_SKIP_AUTH", cast=bool, default=False)

# this will effectively turn off authorization checking,
# allowing for anyone to use the AI functionality
ALLOW_ANONYMOUS_ACCESS = starlette_config("ALLOW_ANONYMOUS_ACCESS", cast=bool, default=False)

logging.info(f"DEBUG is {DEBUG}")
logging.info(f"VERBOSE_INTERNAL_LOGS is {VERBOSE_INTERNAL_LOGS}")

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

PUBLIC_ROUTES = {
    "/",
    "/docs",
    "/docs/",
    "/openapi.json",
    "/openapi.json/",
    "/_status",
    "/_status/",
    "/_version",
    "/_version/",
    "/favicon.ico",
    "/favicon.ico/",
}
ENDPOINTS_WITHOUT_METRICS = {"/metrics", "/metrics/"} | PUBLIC_ROUTES

# This app exports traces using OpenTelemetry. By default in Gen3, we use Alloy for collection.
ENABLE_OPENTELEMETRY_TRACES = starlette_config("ENABLE_OPENTELEMETRY_TRACES", cast=bool, default=True)
# For local development, set this to an EMPTY STRING and it will output to console. See gunicorn.conf.py
OTEL_EXPORTER_OTLP_ENDPOINT = starlette_config(
    "OTEL_EXPORTER_OTLP_ENDPOINT", default="http://alloy.monitoring.4318", cast=str
)

ASYNC_HTTP_CLIENT_TIMEOUT = starlette_config("ASYNC_HTTP_CLIENT_TIMEOUT", cast=float, default=30)

# Metrics provider, at the moment we only support "prometheus". If you want to use a different one,
# you will need to implement the common interface in common/metrics/base.py.
# Note: default is no metrics.
ENABLE_METRICS = starlette_config("ENABLE_METRICS", default=True, cast=bool)
METRICS_PROVIDER = starlette_config("METRICS_PROVIDER", default="prometheus", cast=str)
PROMETHEUS_MULTIPROC_DIR = starlette_config("PROMETHEUS_MULTIPROC_DIR", default="/var/tmp/prometheus_metrics", cast=str)
