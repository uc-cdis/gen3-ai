"""Optional black-box compatibility checks for the Hugging Face client."""

import os
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

huggingface_hub = pytest.importorskip("huggingface_hub")


def test_hf_download_and_model_info_against_running_service(tmp_path):
    """Create an isolated repository and use the real client against it."""
    base_url = os.environ.get("GEN3_AI_MODEL_REPO_TEST_URL")
    if not base_url:
        pytest.skip("set GEN3_AI_MODEL_REPO_TEST_URL to run the live-client test")
    token = os.environ.get("GEN3_AI_MODEL_REPO_TEST_TOKEN")
    debug_skip_auth = os.environ.get("DEBUG_SKIP_AUTH", "false").lower() == "true"
    if not token and not debug_skip_auth:
        pytest.skip("set GEN3_AI_MODEL_REPO_TEST_TOKEN or use the documented DEBUG_SKIP_AUTH=true test mode")

    fixture = Path(__file__).parent / "fixtures/test/repo/config.json"
    fixture_bytes = fixture.read_bytes()
    namespace = "hf-client-test"
    repo = f"repo-{uuid4().hex}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    api = huggingface_hub.HfApi(endpoint=base_url, token=token)

    try:
        with httpx.Client(base_url=base_url, headers=headers) as client:
            response = client.post(
                f"/api/models/{namespace}/{repo}/upload",
                data={"revision_name": "main"},
                files={"files": ("config.json", fixture_bytes, "application/json")},
            )
            response.raise_for_status()

        repo_id = f"{namespace}/{repo}"
        info = api.model_info(repo_id)
        assert info.id == repo_id
        downloaded = huggingface_hub.hf_hub_download(
            repo_id=repo_id,
            filename="config.json",
            revision="main",
            local_dir=str(tmp_path),
            endpoint=base_url,
            token=token,
        )
        assert Path(downloaded).read_bytes() == fixture_bytes
    finally:
        with httpx.Client(base_url=base_url, headers=headers) as client:
            cleanup = client.delete(f"/api/models/{namespace}/{repo}")
            assert cleanup.status_code in {200, 404}
