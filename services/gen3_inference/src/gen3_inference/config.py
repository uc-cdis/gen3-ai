# this is the config module
from common import config as common_config

# this is the starlette.config.Config() class instance
from common.config import starlette_config

# common logger, don't change this
logging = common_config.logging

VERBOSE_LLM_LOGS = starlette_config("VERBOSE_LLM_LOGS", cast=bool, default=False)

logging.info(f"VERBOSE_LLM_LOGS is {VERBOSE_LLM_LOGS}")

URL_PREFIX = starlette_config("GEN3_INFERENCE_PROXY_URL_PREFIX", default="", cast=str)

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
# DO NOT EDIT THE ABOVE
