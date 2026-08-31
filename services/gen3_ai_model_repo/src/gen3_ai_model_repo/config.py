"""Configuration for the Gen3 AI Model Repo service."""

import os

from starlette.datastructures import Secret

from common import config as common_config
from common.config import starlette_config

# common logger, don't change this
logging = common_config.logging

# name of the top-level package in this service
logging.name = "gen3_ai_model_repo"


def _first_env_value(*keys: str, default: str = "") -> str:
    """Return the first non-empty environment variable value from keys."""
    for key in keys:
        value = os.getenv(key)
        if value:
            return value
    return default


MODEL_STORAGE_PATH = starlette_config(
    "MODEL_STORAGE_PATH",
    default="./testfiles",
)


DB_DRIVER = starlette_config(
    "DB_DRIVER",
    default=_first_env_value("PGDRIVER", default="postgresql"),
)

DB_USER = starlette_config(
    "DB_USER",
    default=_first_env_value("PGUSER", default="postgres"),
)

DB_PASSWORD = starlette_config(
    "DB_PASSWORD",
    default=_first_env_value("PGPASSWORD", default="postgres"),
)

DB_HOST = starlette_config(
    "DB_HOST",
    default=_first_env_value("PGHOST", default="localhost"),
)

DB_PORT = starlette_config(
    "DB_PORT",
    cast=int,
    default=_first_env_value("PGPORT", default="5432"),
)

DB_DATABASE = starlette_config(
    "DB_DATABASE",
    default=_first_env_value("PGDATABASE", default="gen3_ai_model_repo"),
)

DB_CONNECTION_STRING = starlette_config(
    "DB_CONNECTION_STRING",
    cast=Secret,
    default=(f"{DB_DRIVER}://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_DATABASE}"),
)

STORAGE_PROVIDER = starlette_config(
    "STORAGE_PROVIDER",
    default="minio",
)

LOCAL_STORAGE_PATH = starlette_config(
    "LOCAL_STORAGE_PATH",
    default="./data",
)

MINIO_ENDPOINT = starlette_config(
    "MINIO_ENDPOINT",
    default="localhost:9000",
)

MINIO_ACCESS_KEY = starlette_config(
    "MINIO_ACCESS_KEY",
    default="minioadmin",
)

MINIO_SECRET_KEY = starlette_config(
    "MINIO_SECRET_KEY",
    default="minioadmin",
)

MINIO_SECURE = starlette_config(
    "MINIO_SECURE",
    cast=bool,
    default=False,
)

MINIO_BUCKET = starlette_config(
    "MINIO_BUCKET",
    default="model-repo",
)


S3_REGION = starlette_config(
    "S3_REGION",
    default="us-east-1",
)

S3_BUCKET = starlette_config(
    "S3_BUCKET",
    default="model-repo",
)

S3_ENDPOINT_URL = starlette_config(
    "S3_ENDPOINT_URL",
    default="",
    cast=str,
)

S3_ACCESS_KEY_ID = starlette_config(
    "S3_ACCESS_KEY_ID",
    default="",
    cast=str,
)

S3_SECRET_ACCESS_KEY = starlette_config(
    "S3_SECRET_ACCESS_KEY",
    default="",
    cast=str,
)

STORAGE_CREATE_BUCKET_IF_MISSING = starlette_config(
    "STORAGE_CREATE_BUCKET_IF_MISSING",
    cast=bool,
    default=True,
)

URL_PREFIX = starlette_config(
    "GEN3_AI_MODEL_REPO_PROXY_URL_PREFIX",
    default=_first_env_value("GEN3_AI_MODEL_REPO_URL_PREFIX", default=""),
    cast=str,
)

# WARNING: Careful changing these, they require close sync with the authorization source
#          of truth. This is the "service" passed to Gen3 Authz for authorization checks
#          and the Authz resource corresponding to the use of the service itself.
#          Additional authorization is applied on a per-AI_MODEL_REPO Resource level within
#          this proxy service, these are a first gate for API-level access. See the
#          rest of the docs/service for more info on AI_MODEL_REPO authz.
AUTHZ_SERVICE_NAME = starlette_config(
    "GEN3_AI_MODEL_REPO_PROXY_AUTHZ_SERVICE_NAME",
    default=_first_env_value("GEN3_AI_MODEL_REPO_AUTHZ_SERVICE_NAME", default="gen3-ai-model-repo"),
    cast=str,
)
AUTHZ_SERVICE_RESOURCE = starlette_config(
    "GEN3_AI_MODEL_REPO_PROXY_AUTHZ_SERVICE_RESOURCE",
    default=_first_env_value("GEN3_AI_MODEL_REPO_AUTHZ_SERVICE_RESOURCE", default="/services/gen3-ai-model-repo"),
    cast=str,
)

FILE_STREAM_CHUNK_SIZE = starlette_config(
    "GEN3_AI_MODEL_REPO_FILE_STREAM_CHUNK_SIZE",
    default=65536,
    cast=int,
)

##### Common Config - DO NOT EDIT #####
# DON'T EDIT THESE *VALUES* IN THIS FILE.
# You can add new common configs here, but do the logic in the common config.
#
# These are here so you can still `import {{SERVICE}}.config`
# and then get `config.{{COMMON_CONFIG}}` in the service code
#
# But the values should be managed by a .env file in the service or ENV VARS
#
# If the common config needs core changes or additions, you need to make the change
# in /libraries/common/src/common/config.py and coordinate updates to the
# services in this section. But bake the logic in the common/config.py, not here.
# Just assign the values here.
DEBUG = common_config.DEBUG
CURRENT_DIR = common_config.CURRENT_DIR
CONFIG_PATH = common_config.CONFIG_PATH
VERBOSE_INTERNAL_LOGS = common_config.VERBOSE_INTERNAL_LOGS
DEBUG_SKIP_AUTH = common_config.DEBUG_SKIP_AUTH
ALLOW_ANONYMOUS_ACCESS = common_config.ALLOW_ANONYMOUS_ACCESS
ARBORIST_URL = common_config.ARBORIST_URL
UNMONITORED_ROUTES = common_config.UNMONITORED_ROUTES
ENDPOINTS_WITHOUT_METRICS = common_config.ENDPOINTS_WITHOUT_METRICS
ENABLE_OPENTELEMETRY_TRACES = common_config.ENABLE_OPENTELEMETRY_TRACES
OTEL_EXPORTER_OTLP_ENDPOINT = common_config.OTEL_EXPORTER_OTLP_ENDPOINT
OTEL_EXPORTER_OTLP_PROTOCOL = common_config.OTEL_EXPORTER_OTLP_PROTOCOL
FORCE_DISABLE_CUSTOM_TRACING = common_config.FORCE_DISABLE_CUSTOM_TRACING
ASYNC_HTTP_CLIENT_TIMEOUT = common_config.ASYNC_HTTP_CLIENT_TIMEOUT
GEN3_AI_MODEL_REPO_URL = common_config.GEN3_AI_MODEL_REPO_URL
GEN3_EMBEDDINGS_URL = common_config.GEN3_EMBEDDINGS_URL
GEN3_INFERENCE_URL = common_config.GEN3_INFERENCE_URL
# DO NOT EDIT THE ABOVE
