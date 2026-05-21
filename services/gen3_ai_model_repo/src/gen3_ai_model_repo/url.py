from urllib.parse import urljoin

DOMAIN = "http://127.0.0.1:4141"


def build_signed_url(namespace: str, repo: str, rev: str, path: str) -> str:
    """Build a signed URL for file retrieval from the mock service."""
    return urljoin(DOMAIN, f"/{namespace}/{repo}/resolve/{rev}/{path}")
