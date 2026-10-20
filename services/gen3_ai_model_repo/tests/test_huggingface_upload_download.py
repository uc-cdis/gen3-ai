"""
Opt-in black-box tests for the Hugging Face client contract.

These tests require a separately running service and a real ``huggingface_hub``
installation.  They deliberately do not replace client calls with handcrafted
HTTP requests.
"""

import hashlib
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import pytest

huggingface_hub = pytest.importorskip("huggingface_hub")


def _client_settings() -> tuple[str, str | None, str]:
    """Return the endpoint, optional token, and namespace for the test run."""
    endpoint = os.environ.get("GEN3_AI_MODEL_REPO_TEST_URL")
    if not endpoint:
        pytest.skip("set GEN3_AI_MODEL_REPO_TEST_URL to a running model-repository service")
    token = os.environ.get("HF_TOKEN") or os.environ.get("GEN3_AI_MODEL_REPO_TEST_TOKEN")
    if not token and os.environ.get("DEBUG_SKIP_AUTH", "false").lower() != "true":
        pytest.skip("set HF_TOKEN or use DEBUG_SKIP_AUTH=true for local testing")
    namespace = os.environ.get("GEN3_AI_MODEL_REPO_TEST_NAMESPACE", "hf-client-test")
    return endpoint.rstrip("/"), token, namespace


def _write_fixture_tree(root: Path, replacement: bytes) -> dict[str, bytes]:
    """Create the model-shaped fixture and return its expected file bytes."""
    expected = {
        "config.json": replacement,
        "model.safetensors": b"test-safetensors-payload",
        "large-model.bin": b"LFS-A" * (11 * 1024 * 1024),
        "tokenizer.json": b'{"version": "1.0"}\n',
        "tokenizer_config.json": b'{"do_lower_case": true}\n',
        "1_Pooling/config.json": b'{"word_embedding_dimension": 8}\n',
    }
    for name, content in expected.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return expected


def _assert_snapshot(snapshot_dir: str, expected: dict[str, bytes]) -> None:
    """Assert that a snapshot has exactly the expected non-empty files."""
    root = Path(snapshot_dir)
    entries = []
    broken_links = []
    for path in root.rglob("*"):
        if path.is_symlink():
            entries.append(f"symlink {path.relative_to(root)} -> {path.resolve(strict=False)}")
            if not path.exists():
                broken_links.append(str(path.relative_to(root)))
        elif path.is_file():
            entries.append(f"file {path.relative_to(root)}")
    actual = {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    if not actual:
        raise AssertionError(
            f"snapshot_download returned {snapshot_dir!r} with no files; "
            f"entries={entries!r}, broken_links={broken_links!r}"
        )
    assert set(actual) == set(expected)
    assert actual == expected
    assert all(content for content in actual.values())


@pytest.mark.integration
def test_huggingface_upload_download_round_trip(tmp_path: Path) -> None:
    """Exercise HfApi repository writes, metadata, replacement, deletion, and snapshots."""
    endpoint, token, namespace = _client_settings()
    api = huggingface_hub.HfApi(endpoint=endpoint, token=token)
    repo_id = f"{namespace}/repo-{uuid4().hex}"
    source = tmp_path / "source"
    expected = _write_fixture_tree(source, b'{"hidden_size": 8}\n')
    created = False

    try:
        repo_url = api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=False, token=token)
        created = True
        assert repo_id in str(repo_url)
        api.upload_file(
            path_or_fileobj=str(source / "config.json"),
            path_in_repo="config.json",
            repo_id=repo_id,
            revision="main",
            token=token,
        )
        api.upload_folder(repo_id=repo_id, folder_path=str(source), revision="main", token=token)

        info = api.model_info(repo_id, revision="main", files_metadata=True, token=token)
        siblings = {sibling.rfilename: sibling for sibling in info.siblings}
        assert set(siblings) == set(expected)
        assert all((sibling.size or 0) > 0 for sibling in siblings.values())
        import httpx

        headers = {"Authorization": f"Bearer {token}"} if token else {}
        with httpx.Client(base_url=endpoint, headers=headers, follow_redirects=False) as client:
            for filename, content in expected.items():
                expected_etag = hashlib.sha256(content).hexdigest()
                resolve_path = f"/{namespace}/{repo_id.split('/', 1)[1]}/resolve/main/{filename}"
                for method in ("head", "get"):
                    response = client.request(method, resolve_path)
                    assert response.status_code in {200, 302}, response.text
                    assert response.headers["x-repo-commit"] == info.sha
                    assert response.headers["x-linked-etag"].strip('"') == expected_etag
                    assert response.headers["x-linked-size"] == str(len(content))
                    if response.status_code == 200:
                        assert response.headers.get("content-length") == str(len(content))
                    if method == "get" and response.status_code == 200:
                        assert response.content == content

        first_cache = tmp_path / "cache-first"
        first_snapshot = huggingface_hub.snapshot_download(
            repo_id=repo_id,
            revision="main",
            cache_dir=str(first_cache),
            endpoint=endpoint,
            token=token,
            force_download=True,
        )
        _assert_snapshot(first_snapshot, expected)
        with ThreadPoolExecutor(max_workers=2) as executor:
            concurrent_snapshots = list(
                executor.map(
                    lambda index: huggingface_hub.snapshot_download(
                        repo_id=repo_id,
                        revision="main",
                        cache_dir=str(tmp_path / f"cache-concurrent-{index}"),
                        endpoint=endpoint,
                        token=token,
                        force_download=True,
                    ),
                    range(2),
                )
            )
        for concurrent_snapshot in concurrent_snapshots:
            _assert_snapshot(concurrent_snapshot, expected)

        replacement = b'{"hidden_size": 16}\n'
        replacement_path = tmp_path / "replacement.json"
        replacement_path.write_bytes(replacement)
        api.upload_file(
            path_or_fileobj=str(replacement_path),
            path_in_repo="config.json",
            repo_id=repo_id,
            revision="main",
            token=token,
        )
        api.delete_file(
            path_in_repo="tokenizer_config.json",
            repo_id=repo_id,
            revision="main",
            commit_message="remove tokenizer config",
            token=token,
        )
        expected["config.json"] = replacement
        del expected["tokenizer_config.json"]

        second_cache = tmp_path / "cache-second"
        second_snapshot = huggingface_hub.snapshot_download(
            repo_id=repo_id,
            revision="main",
            cache_dir=str(second_cache),
            endpoint=endpoint,
            token=token,
            force_download=True,
        )
        _assert_snapshot(second_snapshot, expected)
    finally:
        if created:
            original_error = sys.exc_info()[0]
            try:
                api.delete_repo(repo_id=repo_id, repo_type="model", token=token)
            except BaseException as cleanup_error:
                if original_error is not None:
                    print(f"repository cleanup failed for {repo_id}: {cleanup_error}", file=sys.stderr)
                else:
                    raise AssertionError(f"repository cleanup failed for {repo_id}") from cleanup_error


@pytest.mark.integration
def test_hf_cli_upload_and_download_round_trip(tmp_path: Path) -> None:
    """Exercise the installed ``hf`` executable and reject empty downloads."""
    endpoint, token, namespace = _client_settings()
    hf = shutil.which("hf")
    if not hf:
        pytest.skip("the hf executable is not installed in the test environment")
    api = huggingface_hub.HfApi(endpoint=endpoint, token=token)
    repo_id = f"{namespace}/cli-{uuid4().hex}"
    source = tmp_path / "cli-source"
    expected = _write_fixture_tree(source, b'{"cli": true}\n')
    created = False
    child_env = os.environ.copy()
    child_env["HF_ENDPOINT"] = endpoint
    if token:
        child_env["HF_TOKEN"] = token

    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=False, token=token)
        created = True
        upload = subprocess.run(
            [hf, "upload", repo_id, str(source), "--repo-type", "model", "--revision", "main"],
            env=child_env,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        assert upload.returncode == 0, f"hf upload failed:\n{upload.stdout}\n{upload.stderr}"

        destination = tmp_path / "cli-download"
        download = subprocess.run(
            [hf, "download", repo_id, "--repo-type", "model", "--revision", "main", "--local-dir", str(destination)],
            env=child_env,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        assert download.returncode == 0, f"hf download failed:\n{download.stdout}\n{download.stderr}"
        downloaded = {
            path.relative_to(destination).as_posix(): path.read_bytes()
            for path in destination.rglob("*")
            if path.is_file() and ".cache/huggingface" not in path.relative_to(destination).as_posix()
        }
        assert downloaded, "hf download exited successfully but produced zero files"
        assert downloaded == expected
    finally:
        if created:
            original_error = sys.exc_info()[0]
            try:
                api.delete_repo(repo_id=repo_id, repo_type="model", token=token)
            except BaseException as cleanup_error:
                if original_error is not None:
                    print(f"repository cleanup failed for {repo_id}: {cleanup_error}", file=sys.stderr)
                else:
                    raise AssertionError(f"repository cleanup failed for {repo_id}") from cleanup_error
