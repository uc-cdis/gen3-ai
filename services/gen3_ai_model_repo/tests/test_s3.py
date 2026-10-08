"""Unit tests for the S3 storage provider without a live object store."""

import asyncio
import time
from io import BytesIO
from unittest.mock import MagicMock

import pytest

from gen3_ai_model_repo.storage.s3 import S3StorageProvider


@pytest.fixture
def provider():
    """Build an S3 provider with a mocked client."""
    storage = S3StorageProvider.__new__(S3StorageProvider)
    storage.bucket_name = "test-bucket"
    storage.client = MagicMock()
    storage._upload_concurrency = asyncio.Semaphore(2)
    return storage


@pytest.mark.asyncio
async def test_s3_upload_download_and_metadata(provider, tmp_path):
    """Run upload, download, and metadata operations through the provider."""
    local_file = tmp_path / "downloaded.bin"

    await provider.upload_stream(BytesIO(b"abc123"), "models/demo/weights.bin")
    provider.client.upload_fileobj.assert_called_once()

    await provider.download_file("models/demo/weights.bin", str(local_file))
    provider.client.download_file.assert_called_once_with("test-bucket", "models/demo/weights.bin", str(local_file))

    provider.client.head_object.return_value = {
        "ContentLength": 6,
        "ETag": '"etag"',
        "LastModified": None,
    }
    metadata = await provider.get_file_metadata("models/demo/weights.bin")
    assert metadata == {"size": 6, "etag": "etag", "checksum_sha256": None, "last_modified": None}


@pytest.mark.asyncio
async def test_s3_lists_and_deletes_prefix_in_batches(provider):
    """List objects and delete a prefix using the S3 batch API."""
    page = {"Contents": [{"Key": "models/demo/config.json"}, {"Key": "models/demo/weights.bin"}]}
    provider.client.get_paginator.return_value.paginate.return_value = [page]

    assert await provider.list_files("models/demo") == [
        "models/demo/config.json",
        "models/demo/weights.bin",
    ]

    await provider.delete_prefix("models/demo")
    provider.client.delete_objects.assert_called_once_with(
        Bucket="test-bucket",
        Delete={
            "Objects": [
                {"Key": "models/demo/config.json"},
                {"Key": "models/demo/weights.bin"},
            ]
        },
    )


@pytest.mark.asyncio
async def test_s3_generates_signed_urls(provider):
    """Generate download and upload presigned URLs."""
    provider.client.generate_presigned_url.side_effect = ["download-url", "upload-url"]

    assert await provider.generate_signed_url("models/demo/config.json", expiry_seconds=60) == "download-url"
    assert await provider.generate_upload_url("models/demo/config.json", expiry_seconds=60) == "upload-url"
    assert provider.client.generate_presigned_url.call_count == 2


@pytest.mark.asyncio
async def test_slow_upload_does_not_block_health_work(provider):
    """Allow another coroutine to run while a synchronous upload is slow."""

    def slow_upload(*args, **kwargs):
        del args, kwargs
        time.sleep(0.1)

    provider.client.upload_fileobj.side_effect = slow_upload
    upload_task = asyncio.create_task(provider.upload_stream(BytesIO(b"abc"), "slow.bin"))
    await asyncio.sleep(0)
    health_completed = False
    await asyncio.sleep(0)
    health_completed = True
    assert health_completed
    await upload_task
