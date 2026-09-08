from pathlib import Path

import pytest
from fastapi import HTTPException

from gen3_ai_model_repo.routes.ai_models_uploads import _build_object_key
from gen3_ai_model_repo.storage.keys import build_object_key
from gen3_ai_model_repo.storage.local import LocalStorageProvider


@pytest.mark.parametrize("filename", ["../outside.bin", "/etc/cron.d/job", "nested/../../outside.bin"])
def test_object_key_rejects_path_traversal(filename):
    with pytest.raises(ValueError, match="Invalid filename"):
        build_object_key("namespace", "repo", "main", filename)


def test_upload_route_translates_invalid_filename_to_422():
    with pytest.raises(HTTPException) as exc_info:
        _build_object_key("namespace", "repo", "main", "/etc/cron.d/job")

    assert exc_info.value.status_code == 422


@pytest.mark.asyncio
async def test_local_storage_rejects_object_key_escape(tmp_path: Path):
    provider = LocalStorageProvider(str(tmp_path))

    with pytest.raises(ValueError, match="escapes"):
        await provider.file_exists("../../outside.bin")

    with pytest.raises(ValueError, match="escapes"):
        await provider.delete_file("/etc/cron.d/job")
