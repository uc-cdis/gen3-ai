# this is the config module
from common import config as common_config

# this is the starlette.config.Config() class instance
from common.config import starlette_config

# common logger, don't change this
logging = common_config.logging
logging.name = "gen3_ai_model_repo"

URL_PREFIX = starlette_config("GEN3_AI_MODEL_REPO_PROXY_URL_PREFIX", default="", cast=str)

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
PUBLIC_ROUTES = common_config.PUBLIC_ROUTES
ENDPOINTS_WITHOUT_METRICS = common_config.ENDPOINTS_WITHOUT_METRICS
ENABLE_OPENTELEMETRY_TRACES = common_config.ENABLE_OPENTELEMETRY_TRACES
OTEL_EXPORTER_OTLP_ENDPOINT = common_config.OTEL_EXPORTER_OTLP_ENDPOINT
ASYNC_HTTP_CLIENT_TIMEOUT = common_config.ASYNC_HTTP_CLIENT_TIMEOUT
GEN3_AI_MODEL_REPO_URL = common_config.GEN3_AI_MODEL_REPO_URL
GEN3_EMBEDDINGS_URL = common_config.GEN3_EMBEDDINGS_URL
GEN3_INFERENCE_URL = common_config.GEN3_INFERENCE_URL
# DO NOT EDIT THE ABOVE
