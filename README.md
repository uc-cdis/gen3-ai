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

> Alternative: use a base service instead of library. Justification for library: explicitly labeling the common code as a library (e.g. not meant to be run by itself) provides more clarity on intended use.

Services can import common code from libraries:

```python
from libraries.common import TEST
```

Services have folder structure:

* `src/{{name}}`
* `pyproject.toml` which builds {{name}} from src/{{name}}

> Alternative: all services could share a common `gen3` (or similar) package. Justification to not sharing a common package: the `src` setup is a more explicit separation of concerns and discourages cross-service importing without using the common library.

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

Once you have the above, you can:

```
just setup
just install
just test
```

## Implementation Details

* FastAPI
* PostgreSQL for services that need a database
* No ORM. `asyncpg` with shared code in `libraries/common`
