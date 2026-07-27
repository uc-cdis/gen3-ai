import os

import asyncpg
import pytest
from minio import Minio

from gen3_ai_model_repo.database.db import close_db, connect_db
from gen3_ai_model_repo.storage.helpers import get_storage_provider

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_MODEL_REPO_INTEGRATION_TESTS") != "1",
    reason="Set RUN_MODEL_REPO_INTEGRATION_TESTS=1 and provide live PostgreSQL + MinIO",
)


@pytest.mark.anyio
async def test_storage_provider_minio_end_to_end(tmp_path):
    provider = get_storage_provider()
    await provider.ensure_container()

    sample = tmp_path / "weights.bin"
    sample.write_bytes(b"abc123")

    object_key = "integration/ns/repo/main/weights.bin"
    await provider.upload_file(str(sample), object_key)

    assert await provider.file_exists(object_key)

    downloaded = tmp_path / "downloaded.bin"
    await provider.download_file(object_key, str(downloaded))
    assert downloaded.read_bytes() == b"abc123"

    metadata = await provider.get_file_metadata(object_key)
    assert int(metadata["size"]) == 6

    upload_url = await provider.generate_upload_url(object_key)
    download_url = await provider.generate_signed_url(object_key)
    assert isinstance(upload_url, str) and upload_url
    assert isinstance(download_url, str) and download_url

    await provider.delete_file(object_key)
    assert not await provider.file_exists(object_key)


@pytest.mark.anyio
async def test_postgresql_connectivity_and_migrations_applied():
    await connect_db()
    conn = await asyncpg.connect(
        user=os.getenv("DB_USER", os.getenv("PGUSER", "postgres")),
        password=os.getenv("DB_PASSWORD", os.getenv("PGPASSWORD", "postgres")),
        database=os.getenv("DB_DATABASE", os.getenv("PGDATABASE", "gen3_ai_model_repo")),
        host=os.getenv("DB_HOST", os.getenv("PGHOST", "localhost")),
        port=int(os.getenv("DB_PORT", os.getenv("PGPORT", "5432"))),
    )
    try:
        row = await conn.fetchrow(
            """
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema='public'
              AND table_name='models'
            """
        )
        assert row is not None
    finally:
        await conn.close()
        await close_db()


def test_minio_bucket_exists_for_integration():
    bucket_name = os.getenv("MINIO_BUCKET", "model-repo")
    client = Minio(
        os.getenv("MINIO_ENDPOINT", "localhost:9000"),
        access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        secure=os.getenv("MINIO_SECURE", "false").lower() == "true",
    )
    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)
    assert client.bucket_exists(bucket_name)
