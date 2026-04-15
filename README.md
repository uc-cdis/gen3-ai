# Gen3 AI Monorepo

Contains services for AI features in Gen3.

> Note: The rest of Gen3 does not currently organize itself into a monorepo. We are doing this for these greenfield AI-related features to a) cleanly separate these combined features in one place, b) allow for easier future maintenance and extension of these features, and c) prove out the monorepo idea to see if we can do this in a way that is maintainable and scalable

## API

See [docs/ai_api.yaml](docs/ai_api.yaml) for the OpenAPI specification. You can drop that into [swagger editor](https://editor.swagger.io/) for a nice, interactive API explorer.

## Services Powering the API

- **Gen3 Embeddings**
  - Embeddings and collections of embeddings as first-class objects
  - Bulk support
  - Similarity search over embeddings
  - Row-level authorization
  - Authorization and access control using Gen3 tokens
- **Gen3 Inference**
  - Adopts [Open Responses](https://openresponses.org) standard interface for AI Model inference (both non-streaming and streaming support)
  - Supports mesh-like connections to other Gen3 instances to proxy requests
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
  * The individual services (all import `common`)
* `Dockerfile.k8s`
  * Single Dockerfile with arg `SERVICE` for building different services for a containerized orchestration environment like Kubernetes
* `justfile`
  * Simplified setup, building, running
  * `just setup`, `just install`, `just test`, `just run gen3_embeddings`, `just build`

Services (and libraries) have folder structure:

* `src/{{name}}`
* `pyproject.toml` which builds {{name}} from `src/{{name}}`

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

# verbose output
just lint all -v
just lint services/gen3_inference -v
```

## Implementation Details

* FastAPI
* PostgreSQL for services that need a database
* No ORM. `asyncpg` with shared code in `libraries/common`

### Metrics

See [these docs](./docs/metrics.md) for more info.

## Development Details

### Using VSCode?

See [these docs](./docs/vscode.md) for more info on how to best set up for development of this repo.
