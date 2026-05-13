import json
from datetime import datetime
from pathlib import Path
from shutil import rmtree

MOCK_COMMIT = "mock-commit-hash-123456"


class MetadataService:
    def __init__(self, base_file_dir: Path):
        self.base_file_dir = base_file_dir

    def create_metadata(self, namespace: str, repo: str, description: str, tags: list[str]):
        repo_path = self.base_file_dir / Path(namespace) / Path(repo)
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

    def load_metadata(self, namespace: str, repo: str):
        repo_path = self.base_file_dir / Path(namespace) / Path(repo)
        metadata_file = repo_path / "metadata.json"
        if not metadata_file.is_file():
            raise FileNotFoundError(f"Metadata file not found for {namespace}/{repo}")
        return json.loads(metadata_file.read_text())

    def list_repositories(self):
        repos = []
        for namespace_dir in self.base_file_dir.iterdir():
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

    def delete_repository(self, namespace: str, repo: str):
        repo_path = self.base_file_dir / Path(namespace) / Path(repo)
        if not repo_path.exists():
            raise FileNotFoundError(f"Repository not found: {namespace}/{repo}")
        rmtree(repo_path)
        return True

    def get_revision(self, namespace: str, repo: str, rev: str):
        return {
            "id": f"{namespace}/{repo}",
            "revision": rev,
            "sha": MOCK_COMMIT,
            "commit": MOCK_COMMIT,
            "tags": ["latest", "main"],
        }
