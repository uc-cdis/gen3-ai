from pathlib import Path
from minio import Minio

client = Minio(
    "localhost:9000",
    access_key="minioadmin",
    secret_key="minioadmin",  # pragma: allowlist secret
    secure=False,
)

file_path = Path(__file__).parent / "test.txt"

client.fput_object(
    "model-repo",
    "tests/test.txt",
    str(file_path),
)

print("Upload successful")
