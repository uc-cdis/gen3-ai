import os
from pathlib import Path

import pytest
from minio import Minio


@pytest.mark.skipif(
    os.getenv("RUN_MINIO_INTEGRATION_TESTS") != "1",
    reason="Set RUN_MINIO_INTEGRATION_TESTS=1 with a live MinIO to run this integration test",
)
def test_minio_upload_smoke():
    """Upload a fixture file to MinIO and assert object creation succeeds."""
    client = Minio(
        "localhost:9000",
        access_key="minioadmin",
        secret_key="minioadmin",  # pragma: allowlist secret
        secure=False,
    )

    file_path = Path(__file__).parent / "test.txt"
    result = client.fput_object(
        "model-repo",
        "tests/test.txt",
        str(file_path),
    )

    assert result is not None
