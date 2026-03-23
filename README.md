# Gen3 AI Monorepo

Contains services for AI features in Gen3.

> Note: The rest of Gen3 does not currently organize itself into a monorepo. We are doing this for these greenfield AI-related features to a) cleanly separate these combined features in one place, b) allow for easier future maintenance and extension of these features, and c) prove out the monorepo idea to see if we can do this in a way that is maintainable and scalable

## API

See [docs/ai_api.yaml](docs/ai_api.yaml) for the OpenAPI specification. You can drop that into [swagger editor](https://editor.swagger.io/) for a nice, interactive API explorer.

## Services Powering the API

- **Gen3 Embeddings**
  - Embeddings and vector indicies as first-class objects
  - Row-level authorization
  - Authorization and access control using Gen3 tokens
  - Bulk support
  - Vector search
- **Gen3 Inference**
  - Expose endpoints for model inference and RAG-like interfaces
  - Supports mesh-like connections to other Gen3 instances to share models
  - Highly configurable, can also connect to public cloud services
  - Authorization and access control using Gen3 tokens
- **Gen3 AI Model Repository**
  - AI model management in your own infrastructure (no relying on Huggingface)
  - API exposed is compatible with tools that interact with Huggingface
  - e.g. you can cleanly drop this into existing tools like `transformers` with proper endpoint and creds
  - Authorization and access control using Gen3 tokens

## Layout

* `docs`
  * Additional documentation, including OpenAPI spec
* `libraries/common`
  * The common library and dependencies shared across all services
* `services/{{name}}`
  * The individual services (all import common)
* `Dockerfile`
  * Single Dockerfile with arg `SERVICE` for building different services
* `justfile`
  * Simplified setup, building, running
  * `just install`, `just run gen3_embeddings`, `just build`

Services can import common code:

```python
from common.config import DEBUG
```

Services (and libraries) have folder structure:

* `src/{{name}}`
* `pyproject.toml` which builds {{name}} from src/{{name}}

### Why this setup?

- General benefits of a monorepo (common patterns for maintaining code in a single repo)
- Per-service uv environments ensure minimal required dependencies for each
- Common library project allows cross-service code and dependencies to be
  maintained in one place

## Quickstart

You need config in a `.env` per service.

To get started, we need database info:

`.env` example contents per `/service/{service}` directory:

```text
PGHOST=localhost
PGPORT=5432
PGUSER=postgres
PGPASSWORD=postgres
PGDATABASE={service}
```

You can also configure a global config which overrides common configuration
in a `.env` in the root of the repo:

```
# Common global config for *all* services in monorepo
#
# Intended to be loaded as ENV VARs to override defaults from:
# libraries/common/src/common/config.py
#
# For per-service overrides, supply in /services/{name}/.env
# (which is automatically loaded)

DEBUG=True
```

The DEBUG flag above will override all services configuration when
set as an env var.

Once you have the above `.env`'s, you can:

```
just setup
just install
just test
```

If you want to run a service:

```
just run gen3_embeddings
```

Linting (including formatting):

```bash
# everything
just lint

# specific directory
just lint libraries/common
just lint services/gen3_inference
```


## Implementation Details

* FastAPI
* PostgreSQL for services that need a database
* No ORM. `asyncpg` with shared code in `libraries/common`

## Metrics

By default, we support Prometheus metrics. They can be exposed at a `/metrics` endpoint compatible with Prometheus scraping and visualize in Prometheus or
Graphana, etc.

You can [run Prometheus locally](https://github.com/prometheus/prometheus) if you want to test or visualize these.

### tl;dr

Run the service locally using `just run {{service}}`.

Create a [`prometheus.yml` config file](https://prometheus.io/docs/prometheus/latest/configuration/configuration), such
as: `~/Documents/prometheus/conf/prometheus.yml`.

Put this in:

```yaml
global:
  scrape_interval: 15s # By default, scrape targets every 15 seconds.

# A scrape configuration containing exactly one endpoint to scrape:
# Here it's Prometheus itself.
scrape_configs:
  # The job name is added as a label `job=<job_name>` to any timeseries scraped from this config.
  - job_name: 'gen3_inference'

    # Override the global default and scrape targets from this job every 5 seconds.
    scrape_interval: 10s

    static_configs:
      # NOTE: The `host.docker.internal` below is so docker on MacOS can properly find the locally running service
      - targets: [ 'host.docker.internal:4143' ]

  - job_name: 'gen3_ai_model_repo'
    static_configs:
      - targets: [ 'host.docker.internal:4141' ]
  - job_name: 'gen3_embeddings'
    static_configs:
      - targets: [ 'host.docker.internal:4142' ]
```

> Note: Tested the above config on MacOS, with Linux you can maybe adjust these commands to actually expose the local
> network to the running prometheus container.

Then run this:

```
docker run --name prometheus -v ~/Documents/prometheus/conf/prometheus.yml:/etc/prometheus/prometheus.yml -d -p 127.0.0.1:9090:9090 prom/prometheus
```

Then go to [http://127.0.0.1:9090](http://127.0.0.1:9090).

And some recommended PromQL queries:

```promql
sum by (status_code) (gen3_inference_api_requests_total)
```
