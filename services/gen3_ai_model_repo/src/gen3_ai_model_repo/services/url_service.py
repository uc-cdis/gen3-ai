from urllib.parse import urljoin

# from gen3_ai_model_repo.config import DOMAIN

DOMAIN = "http://127.0.0.1:4141"


class URLService:
    def __init__(self):
        pass

    def build_signed_url(self, namespace: str, repo: str, rev: str, path: str) -> str:
        return urljoin(DOMAIN, f"/{namespace}/{repo}/resolve/{rev}/{path}")
