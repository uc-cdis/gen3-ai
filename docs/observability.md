# Observability

Three signals, three separate paths out of a service:

| Signal      | How it leaves the service | Where it goes                    |
| ----------- | ------------------------- | -------------------------------- |
| **Traces**  | pushed over OTLP          | Alloy (`alloy.monitoring:4318`)  |
| **Metrics** | scraped from `/metrics`   | Prometheus                       |
| **Logs**    | JSON on stdout            | whatever collects container logs |

They are joined by trace ids: `LoggingInstrumentor` puts the active trace and span id on
every log record, and `gen3logging` renders them as `trace_id` / `span_id`. So a log line
found in Loki leads to its trace in Tempo, and vice versa.

## Configuration

| Variable                       | Default                        | Effect                                                                                               |
| ------------------------------ | ------------------------------ | ---------------------------------------------------------------------------------------------------- |
| `ENABLE_OPENTELEMETRY_TRACES`  | `True`                         | Off means no provider, no instrumentation, no spans at all.                                          |
| `FORCE_DISABLE_CUSTOM_TRACING` | `False`                        | On means no per-function spans. Library instrumentation keeps running, so requests and queries are still traced with less detail. It will reduce some CPU cycles for the tracing, but generally that is on the order of <1ms UNLESS you are instrumenting a function that gets called tens of thousands of times or something, in which case it can add up. This config lets you determine if the tracing is what's slowing stuff down, but the solution is NOT to leave it off - you should not trace that function directly (e.g. fix your tracing setup, don't just remove all tracing). |
| `OTEL_EXPORTER_OTLP_ENDPOINT`  | `http://alloy.monitoring:4318` | An **empty string** selects the console exporter, which is how to inspect spans locally without a collector. |
| `OTEL_EXPORTER_OTLP_PROTOCOL`  | `http/protobuf`                | `http/protobuf` pairs with port 4318, `grpc` with 4317. A mismatched pair fails at export time, not at startup. |
| `ENABLE_METRICS`               | `True`                         | Off means `/metrics` serves nothing useful and the counters are never created.                       |
| `METRICS_PROVIDER`             | `prometheus`                   | The only supported value as of now. Anything else with `ENABLE_METRICS=True` fails at startup.       |
| `PROMETHEUS_MULTIPROC_DIR`     | `/var/tmp/prometheus_metrics`  | Directory the per-process counter files live in. **Must exist**, see below.                          |
| `GEN3_JSON_LOGS`               | `True`                         | Off gives human-readable lines with the trace ids appended instead of JSON.                          |

## Traces

`common/telemetry.py` is the only place a tracer provider is installed. Library
instrumentation is automatic: incoming requests (FastAPI), outbound HTTP (httpx and
requests), database queries (asyncpg), and log correlation. On top of that, a service can
emit a span per call for its own functions - see
[Tracing your own functions](#tracing-your-own-functions).

`ENDPOINTS_WITHOUT_METRICS` is excluded from request tracing: the probes (`/_status`,
`/_version`), `/favicon.ico`, and `/metrics` itself.

Everything else is traced, including the site root, `/docs` and `/openapi.json` - browsing the
API is worth seeing.

### Checking traces work

Set `OTEL_EXPORTER_OTLP_ENDPOINT` to an empty string and spans print to stdout, no collector
required:

```bash
OTEL_EXPORTER_OTLP_ENDPOINT= just run gen3_embeddings
```

Hit a real endpoint and you should see, sharing one `trace_id`: the request span, a
`gen3_embeddings.auth.*` span for the authorization work, and a
`gen3_embeddings.database.db.DataAccessLayer.*` span wrapping the asyncpg spans for its
queries. The log lines for that request carry the same `trace_id`.

If you leave the endpoint pointing at a collector that is not reachable, the symptom is
`Failed to export span batch due to timeout, max retries or shutdown.` - the spans were
produced correctly and only the export failed.

### Tracing your own functions

For roughly 10µs per call, a span makes a function's own share of a request's latency
visible. Three ways to add one, in order of preference:

Decorate a function. The span is named `<module>.<qualname>`; exceptions are recorded and
reported as an ERROR span, and still propagate:

```python
from common.telemetry import traced


@traced
async def get_user_authz_mapping(request: Request) -> dict:
    ...
```

Instrument every method a class defines, which is the better option for a class like
`DataAccessLayer` whose methods are all worth tracing. Do this where the app is built, not at
import:

```python
from common.telemetry import instrument_class

instrument_class(DataAccessLayer)
```

Wrap a block inside a function, when the function is too coarse to be useful on its own:

```python
from common.telemetry import get_tracer

tracer = get_tracer(__name__)

with tracer.start_as_current_span("rerank_candidates"):
    ...
```

### What not to trace

**Anything called per row or per loop iteration.** 10µs is nothing once per request and a lot
across 10,000 embeddings. `common.telemetry.no_trace` marks such a function so
`instrument_module` and `instrument_class` skip it:

```python
from common.telemetry import no_trace


@no_trace
def embedding_to_result(row: Record) -> EmbeddingResult:
    ...
```

**Generators and async generators.** A span around one of these ends when the generator
object is created, so it measures nothing, and the wrapper hides the function's
generator-ness from FastAPI's dependency injection. `traced` raises `TypeError` rather than
let that happen, and the walkers skip them. To time the work inside one, put a span around
the body instead.

### `instrument_module` only affects lookups through the module

`instrument_module` rebinds names on the module object. A caller that did
`from x import work` already holds the original function and keeps calling it untraced. That
is why `gen3_embeddings/auth.py` uses `@traced` at each definition: its callers import its
functions by name. Reach for `instrument_module` only when calls go through the module, as in
`x.work()`.

## Metrics

Each service mounts a Prometheus endpoint at `/metrics` as a side effect of building the app,
and counts every request that reaches a metered endpoint from a middleware. Today that is one
counter per service, for example `gen3_embeddings_api_requests_total`, labelled with
`method`, `path`, `status_code` and `user_id`.

`ENDPOINTS_WITHOUT_METRICS` is exempt, so the probes and `/metrics` itself do not inflate the
counts. Documentation traffic is *not* exempt, so `/`, `/docs` and `/openapi.json` each get
their own series.

Two endpoint shapes have no route template for the middleware to label them with, and both are
handled by matching known paths rather than by falling back to the raw URL:

- **`/metrics` is mounted, not routed.** Starlette moves a mount's prefix out of the path and
  into `root_path` before the sub-application sees it, so the request arrives at the middleware
  as `root_path="/metrics"`, `path="/"`. Matching `root_path` too is what keeps a scrape - and
  any subpath under it - exempt.
- **FastAPI's docs and spec are plain Starlette routes**, which never put a route on the scope.
  They are labelled from `app.docs_url` / `app.redoc_url` / `app.openapi_url`.

Anything else that matched no route is labelled `<unmatched>`, so a scanner cannot mint a time
series per URL it tries.

### Keeping cardinality bounded

`path` is the **matched route template**, not the URL that was requested:

```
gen3_embeddings_api_requests_total{method="PATCH",path="/vectorstore/collections/{collection_name}",status_code="401",user_id="Unknown"} 2.0
gen3_embeddings_api_requests_total{method="PATCH",path="<unmatched>",status_code="404",user_id="Unknown"} 1.0
```

Two requests to two *different* collections share one series, which is the point: recording
the concrete path would mint a new time series per collection. A request that matched no route
is recorded as `<unmatched>`, so scanners and typos collapse into one series instead of one
each.

### `PROMETHEUS_MULTIPROC_DIR` must exist

`prometheus_client` chooses between its in-memory and its multiprocess value class **once**,
when it is first imported, based on this variable. Two consequences worth knowing:

- If the directory does not exist, the service dies at startup with
  `ValueError: env PROMETHEUS_MULTIPROC_DIR is not set or not a directory`.
- Anything that sets the variable *after* import leaves counters in memory, where they are
  written nowhere and every scrape comes back empty. This is why the test suite sets it in
  `conftest.py` before importing the app.

On startup you should see it confirmed in the logs:

```
{"logger": "cdispyutils.metrics", "level": "INFO", "message": "PROMETHEUS_MULTIPROC_DIR is /var/tmp/prometheus_metrics", ...}
```

### Checking metrics work

**Fastest: run the tests.** `services/gen3_embeddings/tests/test_metrics.py` covers the whole
path - a served request is counted, a path parameter is recorded as its template, an unrouted
request collapses to one label, an unauthenticated request is counted as an unknown user, and
the exempt endpoints are not counted:

```bash
just test gen3_embeddings
```

**By hand, against a running service.** No Prometheus needed - `/metrics` is just text:

```bash
mkdir -p /var/tmp/prometheus_metrics
just run gen3_embeddings

# in another shell, generate a request and then read the counter
curl -s -X PATCH localhost:8000/vectorstore/collections/my-collection
curl -s localhost:8000/metrics | grep gen3_embeddings_api_requests_total
```

Three things to confirm in that output:

1. A line exists for the request you just made, with the labels you expect.
2. The `path` label is the route template with `{braces}`, not `my-collection`. If you see the
   concrete value, cardinality is broken.
3. Repeat the `curl` against a *different* collection name - the count on that one series
   should go to `2` rather than a second series appearing.

**With a real Prometheus.** See [metrics.md](./metrics.md) for a local Prometheus container,
a scrape config for all three services, and some starting PromQL.

## Logs

Logs go to stdout as one JSON object per line, carrying `timestamp`, `logger`, `level`,
`message`, and the correlation fields `trace_id`, `span_id`, `service`:

```json
{"timestamp": "2026-08-19T18:04:52.162Z", "logger": "cdispyutils.metrics", "level": "INFO", "message": "...", "trace_id": "ac5e71a8c2806eb002e264b525b1253d", "span_id": "b2f936774956355f", "service": "gen3_embeddings"}
```

`trace_id` and `span_id` are `null` outside a request, which is expected for lines emitted at
import or startup. Inside a request they match the span in Tempo, which is the whole reason
`LoggingInstrumentor` is installed.

Set `GEN3_JSON_LOGS=False` for human-readable lines while developing.
