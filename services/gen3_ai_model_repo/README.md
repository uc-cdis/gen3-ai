# Gen3 AI Model Repository Service

A FastAPI-based service for managing AI model repositories with metadata tracking, file versioning, and secure file access. This service implements a subset of the Hugging Face Hub API for model discovery and retrieval.

## Overview

The Gen3 AI Model Repository Service provides:

- **Repository Management**: Create, list, and delete model repositories organized by namespace
- **File Tracking**: Track all files in repositories with SHA-256 content hashing
- **Revision Management**: Support multiple revisions (versions) of repositories
- **Metadata Storage**: Store and retrieve model descriptions, tags, and creation timestamps
- **Secure File Access**: Generate signed URLs for secure file downloads
- **Health Monitoring**: Built-in health check endpoint

## Architecture

### Components

- **Routes** (`routes/ai_models.py`): FastAPI endpoint definitions for model management
- **Storage** (`storage.py`): Local file system operations and hash computation
- **Metadata** (`metadata.py`): Repository metadata file operations
- **Database** (`database/`): PostgreSQL integration for persistent metadata
- **Models** (`models/schemas.py`): Pydantic schema definitions for API contracts

### Database Schema

The service uses PostgreSQL with the following tables:

- `model_repositories`: Stores repository metadata (namespace, name, description, tags)
- `model_revisions`: Tracks different versions/revisions of repositories
- `model_files`: Tracks individual files within revisions with content hashes

All tables include appropriate indexes for efficient querying.

## Installation

### Prerequisites

- Python 3.13+
- PostgreSQL 12+
- UV (Python package manager)

### Setup

1. **Install dependencies**:
   ```bash
   cd services/gen3_ai_model_repo
   uv pip install -e .
   ```

2. **Initialize the database**:
   ```bash
   # Create the database
   createdb gen3_model_repo

   # Apply migrations
   psql -d gen3_model_repo -f src/gen3_ai_model_repo/db_migrations/001_initial_schema.sql
   ```

3. **Set up environment variables**:
   ```bash
   export DB_HOST=localhost
   export DB_PORT=5432
   export DB_NAME=gen3_model_repo
   export DB_USER=your_db_user
   export DB_PASSWORD=your_db_password
   export DEBUG=false
   export AUTH_REQUIRED=true  # Requires valid Bearer tokens
   ```

## Running the Service

### Development

```bash
# Start the development server with auto-reload
uvicorn gen3_ai_model_repo.main:app --reload --host 0.0.0.0 --port 8000
```

The API documentation will be available at: `http://localhost:8000/docs`

### Production

```bash
# Using gunicorn with multiple workers
gunicorn gen3_ai_model_repo.main:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000
```

## API Endpoints

All endpoints are documented with OpenAPI/Swagger specs. Visit `/docs` for interactive documentation.

### Model Repository Operations

#### Upload Model
- **POST** `/api/models/{namespace}/{repo}/upload`
- Upload a new model repository with metadata
- Requires authentication
- Returns: `UploadModelResponse` with repository info

#### List Models
- **GET** `/api/models`
- Retrieve all available repositories
- Returns: List of `RepositoryModel` objects

#### Get Model Info
- **GET** `/api/models/{namespace}/{repo}/info`
- Get comprehensive repository information including metadata and files
- Returns: `RepositoryInfoModel` with full repository details

#### Delete Model
- **DELETE** `/api/models/{namespace}/{repo}`
- Delete a repository and all its files
- Requires authentication
- Returns: `DeleteModelResponse`

### File Operations

#### Get File (with redirect)
- **GET** `/api/models/{namespace}/{repo}/resolve/{rev}/{path:path}`
- Get download redirect to signed URL
- Returns: HTTP 302 redirect to file download endpoint

#### Get File Metadata (HEAD)
- **HEAD** `/api/models/{namespace}/{repo}/resolve/{rev}/{path:path}`
- Get file metadata without downloading
- Returns: File size, content hashes, signed URL in headers

#### Stream File
- **GET** `/signed-url/{path:path}`
- Stream file content with proper Content-Length header
- Returns: File content streamed in 64KB chunks

### Repository Structure

#### List Directory
- **GET** `/api/models/{namespace}/{repo}/tree/{rev}`
- **GET** `/api/models/{namespace}/{repo}/tree/{rev}/{path:path}`
- List directory contents or file information
- Returns: List of `TreeEntryModel` objects

#### Get Revision Info
- **GET** `/api/models/{namespace}/{repo}/revision/{rev}`
- Get metadata for a specific revision
- Returns: `RevisionModel` with commit SHA and tags

#### List Revisions
- **GET** `/api/models/{namespace}/{repo}/revisions`
- List all revisions of a repository
- Requires authentication
- Returns: `RevisionListResponseModel` with revision history

### System Operations

#### Health Check
- **GET** `/health`
- Check service health and readiness
- Returns: HTTP 200 OK

## Authentication

The service uses Bearer token authentication for protected endpoints. Include the token in the `Authorization` header:

```bash
curl -H "Authorization: Bearer your-token" \
  https://api.example.com/api/models
```

Protected endpoints:
- POST `/api/models/{namespace}/{repo}/upload`
- DELETE `/api/models/{namespace}/{repo}`
- GET `/api/models/{namespace}/{repo}/revisions`

## Database Schema and Migrations

### Single Consolidated Migration

The service uses a single migration file: `001_initial_schema.sql`

This migration creates:
- `model_repositories` table with indexes
- `model_revisions` table with triggers
- `model_files` table with content hash tracking
- Automatic `updated_at` timestamp triggers

To apply or reset migrations:

```bash
# Apply migrations
psql -d gen3_model_repo -f src/gen3_ai_model_repo/db_migrations/001_initial_schema.sql

# Reset database (WARNING: deletes all data)
dropdb gen3_model_repo
createdb gen3_model_repo
psql -d gen3_model_repo -f src/gen3_ai_model_repo/db_migrations/001_initial_schema.sql
```

## File Storage

### Current Implementation

Currently, files are stored on the local file system in the `routes/testfiles/` directory. The directory structure is:

```
testfiles/
└── {namespace}/
    └── {repo}/
        ├── metadata.json
        └── [model files]
```

### S3 Integration (Production)

In production, the service should integrate with AWS S3 for file storage:

1. **File Upload**: Store uploaded files in S3 buckets organized by namespace
2. **Signed URLs**: Generate S3 pre-signed URLs for secure downloads
3. **Cost Optimization**: Use S3's features (lifecycle policies, intelligent tiering) for cost management
4. **Security**: Leverage S3 bucket policies and IAM roles for access control

**Key Changes Required**:
- Replace local file operations in `storage.py` with boto3 S3 operations
- Update `get_file()` endpoint to generate S3 signed URLs
- Implement S3 bucket configuration and lifecycle policies
- Add S3 error handling and retry logic

**Implementation Steps**:
```python
# Example S3 integration pattern
import boto3

s3_client = boto3.client('s3')
bucket_name = os.environ.get('S3_BUCKET_NAME')

# Generate signed URL
url = s3_client.generate_presigned_url(
    'get_object',
    Params={'Bucket': bucket_name, 'Key': f'{namespace}/{repo}/{path}'},
    ExpiresIn=3600  # 1 hour expiration
)
```

## Testing

Run tests with uv (Python 3.13 environment):

```bash
# Run unit tests (exclude MinIO integration-style module)
uv run --with pytest --with pytest-asyncio pytest -q tests --ignore tests/test_minio.py

# Run with coverage
uv run --with pytest --with pytest-asyncio --with pytest-cov \
    pytest --cov=gen3_ai_model_repo tests --ignore tests/test_minio.py

# Run a specific test file
uv run --with pytest --with pytest-asyncio pytest -q tests/test_routes.py


MinIO note:

- tests/test_minio.py performs a live upload to MinIO at localhost:9000 during module import.
- Keep MinIO running before including that file in test runs, or exclude it with --ignore tests/test_minio.py for unit-only runs.
# Run with verbose output
pytest -v tests/
```

### Test Coverage

The test suite includes:
- Health endpoint tests
- Repository listing tests
- Storage utility function tests
- Metadata CRUD operations
- Integration tests for complete workflows
- Error handling and edge cases

Current test categories:
- **TestHealthEndpoint**: Service health monitoring
- **TestListModelsEndpoint**: Repository listing
- **TestStorageFunctions**: Hash computation and file operations
- **TestMetadataFunctions**: Metadata file operations
- **TestErrorHandling**: HTTP error responses
- **TestIntegration**: End-to-end workflows
- **TestUploadModelRequest**: Request validation

## Configuration

Environment variables for service configuration:

| Variable | Default | Description |
|----------|---------|-------------|
| `DEBUG` | `false` | Enable FastAPI debug mode |
| `DB_HOST` | `localhost` | PostgreSQL server host |
| `DB_PORT` | `5432` | PostgreSQL server port |
| `DB_NAME` | `gen3_model_repo` | Database name |
| `DB_USER` | `` | Database user |
| `DB_PASSWORD` | `` | Database password |
| `AUTH_REQUIRED` | `true` | Require authentication for all endpoints |
| `URL_PREFIX` | `/` | URL prefix for the service |
| `PORT` | `8000` | Service port |

## Development

### Code Style

- Uses `ruff` for linting and formatting
- Google-style docstrings with Args/Returns sections
- Type hints on all functions

### Running Linter

```bash
# Check code style
ruff check .

# Fix style issues
ruff format .
```

### Docstring Format

All functions use Google-style docstrings:

```python
def my_function(param1: str, param2: int) -> str:
    """
    Brief description of what the function does.

    Longer description explaining the function in more detail,
    including any important behaviors or side effects.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of return value.

    Raises:
        ValueError: When invalid input is provided.
    """
```

## OpenAPI Specification

The service generates an OpenAPI 3.0 specification automatically. Access it at:

- Interactive Docs: `http://localhost:8000/docs` (Swagger UI)
- ReDoc: `http://localhost:8000/redoc`
- JSON Schema: `http://localhost:8000/openapi.json`

All endpoints include:
- Comprehensive descriptions
- Parameter documentation
- Response models with schemas
- HTTP status code definitions
- Error response documentation

## Troubleshooting

### Database Connection Failed

```
StartupException: Failed database connection test
```

**Solution**: Check PostgreSQL is running and credentials are correct:
```bash
psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c "SELECT 1"
```

### Port Already in Use

```
OSError: [Errno 48] Address already in use
```

**Solution**: Use a different port:
```bash
uvicorn gen3_ai_model_repo.main:app --port 8001
```

### Import Errors

**Solution**: Ensure package is installed in development mode:
```bash
uv pip install -e .
```

### Authentication Failures

**Solution**: Verify Bearer token format in requests:
```bash
# Correct
curl -H "Authorization: Bearer my-token"

# Incorrect
curl -H "Authorization: my-token"
```

## Performance Considerations

- **Hash Computation**: Files are hashed using SHA-256. Large files will have a computation overhead.
- **Database Indexes**: Queries on namespace, repo, and file paths benefit from indexes.
- **File Streaming**: Large files are streamed in 64KB chunks to avoid memory issues.
- **Connection Pooling**: Database connections are pooled for efficiency.

## Security

- **Authentication**: Bearer token validation for sensitive operations
- **Path Validation**: File paths are validated to prevent directory traversal
- **CORS**: Configure CORS policies for cross-origin requests in production
- **HTTPS**: Use HTTPS in production for encrypted communication

## Contributing

When contributing to this service:

1. Add Google-style docstrings to all functions
2. Write unit tests for new functionality
3. Update API documentation in endpoint decorators
4. Run tests and linting before committing
5. Update this README with any new features

## License

See LICENSE file in the repository root.

## Related Services

- **gen3_embeddings**: Vector embedding generation service
- **gen3_inference**: LLM inference service using uploaded models
- **common**: Shared utilities and authentication
