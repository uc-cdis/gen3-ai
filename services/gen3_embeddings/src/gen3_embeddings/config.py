"""
Configuration handling, should use common config and overlay service-specific
configuration.
"""

from starlette.datastructures import Secret

from common import config as common_config

# this is the starlette.config.Config() class instance
from common.config import starlette_config

# common logger, don't change this
logging = common_config.logging

# name of the top-level package in this service
logging.name = "gen3_embeddings"

# gunicorn setting for the number of workers to spawn, see https://docs.gunicorn.org/en/stable/settings.html#workers
GUNICORN_WORKERS = starlette_config("GUNICORN_WORKERS", cast=int, default="1")

DEFAULT_PAGE_SIZE = starlette_config("DEFAULT_PAGE_SIZE", cast=int, default=100)
MAX_PAGE_SIZE = starlette_config("MAX_PAGE_SIZE", cast=int, default=1000)

##### Request Limits #####
# Upper bounds on the caller-controlled inputs whose cost to this service scales with
# whatever the caller sent. Without them a single well-formed request can consume memory
# or CPU without limit, so these are a denial-of-service control, not input hygiene.

MAX_REQUEST_BODY_BYTES = starlette_config("MAX_REQUEST_BODY_BYTES", cast=int, default=64 * 1024 * 1024)

# Bounds on a single vector. MAX_VECTOR_DIMENSIONS also caps `dimensions` at collection
# creation, which is what every later embedding in that collection is validated against.
MIN_VECTOR_DIMENSIONS = starlette_config("MIN_VECTOR_DIMENSIONS", cast=int, default=1)
MAX_VECTOR_DIMENSIONS = starlette_config("MAX_VECTOR_DIMENSIONS", cast=int, default=8192)

# Bounds on the raw-text inputs. Embedding text is not implemented yet, so both of the
# fields these apply to currently end in a 4xx, but an unbounded field costs the memory to
# parse it whether or not the handler goes on to use it.
MAX_TEXT_INPUT_LENGTH = starlette_config("MAX_TEXT_INPUT_LENGTH", cast=int, default=32 * 1024)
MAX_TEXT_CHUNKS = starlette_config("MAX_TEXT_CHUNKS", cast=int, default=256)

# How many embeddings one create/update request may carry. Cost per item is a vector
# parse, a JSON re-serialization, and a row write, so this multiplies against
# MAX_VECTOR_DIMENSIONS.
MAX_EMBEDDINGS_PER_REQUEST = starlette_config("MAX_EMBEDDINGS_PER_REQUEST", cast=int, default=1000)

# How many UUIDs one bulk read may ask for. These requests are tiny (36 bytes per UUID)
# but each one returns a full vector, so the response amplification is what is bounded
# here rather than the request.
MAX_EMBEDDING_UUIDS_PER_REQUEST = starlette_config("MAX_EMBEDDING_UUIDS_PER_REQUEST", cast=int, default=1000)

# Per-embedding metadata. Stored as jsonb and hashed on write, and returned in full on
# every read of that embedding, so it is bounded by serialized size rather than key count
# alone. MAX_METADATA_DEPTH bounds nesting, which costs stack rather than bytes.
MAX_METADATA_BYTES = starlette_config("MAX_METADATA_BYTES", cast=int, default=64 * 1024)
MAX_METADATA_KEYS = starlette_config("MAX_METADATA_KEYS", cast=int, default=256)
MAX_METADATA_DEPTH = starlette_config("MAX_METADATA_DEPTH", cast=int, default=16)

# Free-text fields that end up in a database column or an authz resource path.
MAX_COLLECTION_NAME_LENGTH = starlette_config("MAX_COLLECTION_NAME_LENGTH", cast=int, default=255)
MAX_DESCRIPTION_LENGTH = starlette_config("MAX_DESCRIPTION_LENGTH", cast=int, default=4096)
MAX_AUTHZ_LENGTH = starlette_config("MAX_AUTHZ_LENGTH", cast=int, default=1024)

# The `ai_model` query parameter. Not wired up to anything yet, but it is caller-controlled
# free text that reaches the log lines and, eventually, the model service.
MAX_AI_MODEL_NAME_LENGTH = starlette_config("MAX_AI_MODEL_NAME_LENGTH", cast=int, default=256)

# Search bounds. `top_k` becomes a SQL LIMIT over rows that each carry a full vector, so
# an unbounded value returns the whole table. Each filter adds a WHERE clause and two
# query parameters to a generated statement.
MAX_TOP_K = starlette_config("MAX_TOP_K", cast=int, default=1000)
MAX_SEARCH_FILTERS = starlette_config("MAX_SEARCH_FILTERS", cast=int, default=20)
MAX_SEARCH_FILTER_KEY_LENGTH = starlette_config("MAX_SEARCH_FILTER_KEY_LENGTH", cast=int, default=256)
MAX_SEARCH_FILTER_VALUE_LENGTH = starlette_config("MAX_SEARCH_FILTER_VALUE_LENGTH", cast=int, default=1024)

# How many names the `collections` query parameter on cross-collection search may name.
# Each one currently costs its own database round trip.
MAX_COLLECTIONS_PER_SEARCH = starlette_config("MAX_COLLECTIONS_PER_SEARCH", cast=int, default=100)

# How many collections one cross-collection search may fan out over when the caller does
# not name any, and we search everything they are authorized for. Deliberately larger than
# MAX_COLLECTIONS_PER_SEARCH: that bound counts round trips, one per name, while this one
# counts collection_ids in the single ANY(...) search query, which is a much cheaper unit.
# Past this the caller has to name collections explicitly, which loses nothing - search
# ranks by a per-row metric, so the top_k of a union is the top_k of the per-batch top_ks.
MAX_COLLECTIONS_SEARCHED = starlette_config("MAX_COLLECTIONS_SEARCHED", cast=int, default=1000)

# Length of that same parameter before it is split. Derived rather than configured, because
# the only thing it needs to be is wide enough for MAX_COLLECTIONS_PER_SEARCH names of the
# longest permitted length, plus their separating commas. Splitting a string is what costs
# here, so it has to be bounded before the split rather than after.
MAX_COLLECTIONS_QUERY_LENGTH = MAX_COLLECTIONS_PER_SEARCH * (MAX_COLLECTION_NAME_LENGTH + 1)

# Deepest page a caller may request. Bounds the OFFSET we hand Postgres, which is an
# int4, and avoids serving arbitrarily deep scans. Paging past this needs a narrower
# query, not a bigger offset.
MAX_PAGE = starlette_config("MAX_PAGE", cast=int, default=10_000)


PGDRIVER = starlette_config("PGDRIVER", default="postgresql")
PGUSER = starlette_config("PGUSER", default="postgres")
PGPASSWORD = starlette_config("PGPASSWORD", cast=Secret, default=None)
PGHOST = starlette_config("PGHOST", default="localhost")
PGPORT = starlette_config("PGPORT", cast=int, default="5432")
PGDATABASE = starlette_config("PGDATABASE", default="gen3embeddings")

PGPOOL_MIN_SIZE = starlette_config("PGPOOL_MIN_SIZE", cast=int, default="1")
PGPOOL_MAX_SIZE = starlette_config("PGPOOL_MAX_SIZE", cast=int, default="5")

DB_CONNECTION_STRING = starlette_config(
    "DB_CONNECTION_STRING",
    cast=Secret,
    default=f"{PGDRIVER}://{PGUSER}:{PGPASSWORD}@{PGHOST}:{PGPORT}/{PGDATABASE}",
)

URL_PREFIX = starlette_config("GEN3_EMBEDDINGS_URL_PREFIX", default="", cast=str)

# WARNING: Careful changing these, they require close sync with the authorization source
#          of truth. This is the "service" passed to Gen3 Authz for authorization checks
#          and the Authz resource corresponding to the use of the service itself.
#          Additional authorization is applied on a per-EMBEDDINGS Resource level within
#          this proxy service, these are a first gate for API-level access. See the
#          rest of the docs/service for more info on EMBEDDINGS authz.
AUTHZ_SERVICE_NAME = starlette_config("GEN3_EMBEDDINGS_AUTHZ_SERVICE_NAME", default="gen3-embeddings", cast=str)
# AUTHZ_SERVICE_RESOURCE = starlette_config(
#    "GEN3_EMBEDDINGS_AUTHZ_SERVICE_RESOURCE",
#    default="/services/gen3-embeddings",
#    cast=str,
# )

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
