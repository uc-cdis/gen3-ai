import json
from datetime import datetime
from pathlib import Path
from shutil import rmtree
from typing import Any

MOCK_COMMIT = "mock-commit-hash-123456"


def create_metadata(
    base_file_dir: Path,
    namespace: str,
    repo: str,
    description: str,
    tags: list[str],
) -> Path:
    """Create repository metadata JSON for a repo namespace and return the file path."""
    repo_path = base_file_dir / Path(namespace) / Path(repo)
    repo_path.mkdir(parents=True, exist_ok=True)
    metadata_file = repo_path / "metadata.json"

    metadata_content = {
        "namespace": namespace,
        "repo": repo,
        "description": description,
        "tags": tags,
        "created_at": datetime.now().isoformat(timespec="seconds") + "Z",
    }
    metadata_file.write_text(json.dumps(metadata_content))
    return metadata_file


def load_metadata(base_file_dir: Path, namespace: str, repo: str) -> dict[str, Any]:
    """Load metadata JSON for a given repository."""
    repo_path = base_file_dir / Path(namespace) / Path(repo)
    metadata_file = repo_path / "metadata.json"
    if not metadata_file.is_file():
        raise FileNotFoundError(f"Metadata file not found for {namespace}/{repo}")
    return json.loads(metadata_file.read_text())


def list_repositories(base_file_dir: Path) -> list[dict[str, Any]]:
    """Return a list of mocked repository metadata summaries."""
    repos: list[dict[str, Any]] = []
    for namespace_dir in base_file_dir.iterdir():
        if namespace_dir.is_dir():
            for repo_dir in namespace_dir.iterdir():
                if repo_dir.is_dir():
                    repos.append(
                        {
                            "id": f"{namespace_dir.name}/{repo_dir.name}",
                            "description": "This is a mock model repository.",
                            "tags": ["example", "test"],
                            "created_at": datetime.now().isoformat(timespec="seconds") + "Z",
                        }
                    )
    return repos


def delete_repository(base_file_dir: Path, namespace: str, repo: str) -> bool:
    """Delete a repository directory and its contents."""
    repo_path = base_file_dir / Path(namespace) / Path(repo)
    if not repo_path.exists():
        raise FileNotFoundError(f"Repository not found: {namespace}/{repo}")
    rmtree(repo_path)
    return True


def repository_exists(base_file_dir: Path, namespace: str, repo: str) -> bool:
    """Return whether a repository exists on disk."""
    repo_path = base_file_dir / Path(namespace) / Path(repo)
    return repo_path.exists() and repo_path.is_dir()


def get_revision(namespace: str, repo: str, rev: str) -> dict[str, Any]:
    """Return mock revision metadata for a requested revision."""
    return {
        "id": f"{namespace}/{repo}",
        "revision": rev,
        "sha": MOCK_COMMIT,
        "commit": MOCK_COMMIT,
        "tags": ["latest", "main"],
    }
