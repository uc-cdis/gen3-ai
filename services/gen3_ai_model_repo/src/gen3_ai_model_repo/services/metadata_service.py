MOCK_COMMIT = "mock-commit-hash-123456"


class MetadataService:
    def __init__(self):
        pass

    def get_revision(self, namespace: str, repo: str, rev: str):
        return {
            "id": f"{namespace}/{repo}",
            "revision": rev,
            "sha": MOCK_COMMIT,
            "commit": MOCK_COMMIT,
            "tags": ["latest", "main"],
        }
