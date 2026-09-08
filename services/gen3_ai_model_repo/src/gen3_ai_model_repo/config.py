"""Configuration for the Gen3 AI Model Repo service."""

from starlette.datastructures import Secret

from common import config as common_config
from common.config import starlette_config

# common logger, don't change this
logging = common_config.logging

# name of the top-level package in this service
logging.name = "gen3_ai_model_repo"


PGDRIVER = starlette_config("PGDRIVER", default="postgresql")
PGUSER = starlette_config("PGUSER", default="postgres")
PGPASSWORD = starlette_config("PGPASSWORD", cast=Secret, default="postgres")
PGHOST = starlette_config("PGHOST", default="localhost")
PGPORT = int(starlette_config("PGPORT", cast=int, default="5432"))
PGDATABASE = starlette_config("PGDATABASE", default="gen3_ai_model_repo")
PGPOOL_MIN_SIZE = int(starlette_config("PGPOOL_MIN_SIZE", cast=int, default="1"))
PGPOOL_MAX_SIZE = int(starlette_config("PGPOOL_MAX_SIZE", cast=int, default="5"))
DEFAULT_PAGE_SIZE = int(starlette_config("DEFAULT_PAGE_SIZE", cast=int, default="100"))
MAX_PAGE_SIZE = int(starlette_config("MAX_PAGE_SIZE", cast=int, default="1000"))
MAX_SEARCH_LENGTH = int(starlette_config("MAX_SEARCH_LENGTH", cast=int, default="256"))
MAX_UPLOAD_FILES = int(starlette_config("MAX_UPLOAD_FILES", cast=int, default="100"))
MAX_UPLOAD_BYTES = int(starlette_config("MAX_UPLOAD_BYTES", cast=int, default="5368709120"))

DB_CONNECTION_STRING = starlette_config(
    "DB_CONNECTION_STRING",
    cast=Secret,
    default=f"{PGDRIVER}://{PGUSER}:{PGPASSWORD}@{PGHOST}:{PGPORT}/{PGDATABASE}",
)

STORAGE_PROVIDER = starlette_config("STORAGE_PROVIDER", default="local")

LOCAL_STORAGE_PATH = starlette_config(
    "LOCAL_STORAGE_PATH",
    default="./data",
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

S3_SESSION_TOKEN = starlette_config(
    "S3_SESSION_TOKEN",
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
    default="",
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
    default="gen3-ai-model-repo",
    cast=str,
)
AUTHZ_SERVICE_RESOURCE = starlette_config(
    "GEN3_AI_MODEL_REPO_PROXY_AUTHZ_SERVICE_RESOURCE",
    default="/services/gen3-ai-model-repo",
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
