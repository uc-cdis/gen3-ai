import os

import asyncpg
import pytest

from gen3_ai_model_repo.database.db import close_db, connect_db
from gen3_ai_model_repo.storage.helpers import get_storage_provider

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_MODEL_REPO_INTEGRATION_TESTS") != "1",
    reason="Set RUN_MODEL_REPO_INTEGRATION_TESTS=1 and provide live PostgreSQL + configured object storage",
)


@pytest.mark.asyncio
async def test_storage_provider_end_to_end(tmp_path):
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


@pytest.mark.asyncio
async def test_postgresql_connectivity_and_migrations_applied():
    await connect_db()
    conn = await asyncpg.connect(
        user=os.getenv("PGUSER", "postgres"),
        password=os.getenv("PGPASSWORD", "postgres"),
        database=os.getenv("PGDATABASE", "gen3_ai_model_repo"),
        host=os.getenv("PGHOST", "localhost"),
        port=int(os.getenv("PGPORT", "5432")),
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
