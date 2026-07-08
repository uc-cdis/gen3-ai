from urllib.parse import urljoin

from gen3_ai_model_repo.config import GEN3_AI_MODEL_REPO_URL


def build_signed_url(namespace: str, repo: str, rev: str, path: str) -> str:
    """Build a signed URL for file retrieval from the configured service URL."""
    return urljoin(str(GEN3_AI_MODEL_REPO_URL), f"/{namespace}/{repo}/resolve/{rev}/{path}")
