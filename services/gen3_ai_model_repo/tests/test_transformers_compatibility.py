"""Opt-in black-box test for the documented Transformers integration."""

import base64
import hashlib
import os
from pathlib import Path
from uuid import uuid4

import pytest


@pytest.mark.integration
def test_transformers_from_pretrained_against_model_repo(tmp_path, monkeypatch):
    """Create a tiny local model, upload it, and use real client libraries."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    import httpx

    huggingface_hub = pytest.importorskip("huggingface_hub")

    endpoint = os.environ.get("HF_ENDPOINT") or os.environ.get("GEN3_AI_MODEL_REPO_TEST_URL")
    if not endpoint:
        pytest.skip("set HF_ENDPOINT to a running model-repository service")
    token = os.environ.get("GEN3_AI_MODEL_REPO_TEST_TOKEN")
    if not token and os.environ.get("DEBUG_SKIP_AUTH", "false").lower() != "true":
        pytest.skip("set GEN3_AI_MODEL_REPO_TEST_TOKEN or use DEBUG_SKIP_AUTH=true for local testing")
    monkeypatch.setenv("HF_ENDPOINT", endpoint)

    vocab = tmp_path / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\n")
    tokenizer = transformers.BertTokenizerFast(vocab_file=str(vocab))
    model = transformers.BertModel(
        transformers.BertConfig(
            vocab_size=7, hidden_size=8, num_hidden_layers=1, num_attention_heads=2, intermediate_size=16
        )
    )
    artifact_dir = tmp_path / "artifact"
    tokenizer.save_pretrained(artifact_dir)
    model.save_pretrained(artifact_dir, safe_serialization=True)
    files = [path for path in artifact_dir.iterdir() if path.is_file()]

    namespace = "transformers-test"
    repo = f"repo-{uuid4().hex}"
    repo_id = f"{namespace}/{repo}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        with httpx.Client(base_url=endpoint, headers=headers) as api_client:
            response = api_client.post(f"/api/models/{repo_id}", json={"description": "compatibility test"})
            response.raise_for_status()
            intent_ids = []
            for path in files:
                checksum = base64.b64encode(hashlib.sha256(path.read_bytes()).digest()).decode()
                intent_response = api_client.post(
                    f"/api/models/{repo_id}/upload-url",
                    json={
                        "revision_name": "main",
                        "file_name": path.name,
                        "checksum_sha256": checksum,
                        "file_size": path.stat().st_size,
                    },
                )
                intent_response.raise_for_status()
                intent = intent_response.json()
                assert intent["headers"].get("x-amz-checksum-sha256") == checksum
                with httpx.Client() as storage_client:
                    upload = storage_client.put(
                        intent["upload_url"],
                        headers=intent["headers"],
                        content=path.read_bytes(),
                    )
                if upload.status_code >= 400:
                    raise AssertionError(
                        f"storage PUT failed with status {upload.status_code}: {upload.text}; "
                        f"sent headers: {sorted(intent['headers'])}"
                    )
                intent_ids.append(intent["upload_id"])
            complete = api_client.post(
                f"/api/models/{repo_id}/complete-upload",
                json={"revision_name": "main", "upload_ids": intent_ids},
            )
            complete.raise_for_status()

            api = huggingface_hub.HfApi(endpoint=endpoint, token=token)
            info = api.model_info(repo_id)
            assert info.id == repo_id
            downloaded = huggingface_hub.hf_hub_download(
                repo_id=repo_id,
                filename="config.json",
                revision="main",
                local_dir=str(tmp_path / "download"),
                endpoint=endpoint,
                token=token,
            )
            assert Path(downloaded).read_bytes()
            loaded_tokenizer = transformers.AutoTokenizer.from_pretrained(repo_id, revision="main", token=token)
            loaded_model = transformers.AutoModel.from_pretrained(repo_id, revision="main", token=token)
            result = loaded_model(**loaded_tokenizer("hello world", return_tensors="pt"))
            assert result.last_hidden_state.shape[0] == 1
            assert torch.isfinite(result.last_hidden_state).all()
    finally:
        with httpx.Client(base_url=endpoint, headers=headers) as client:
            client.delete(f"/api/models/{repo_id}")
