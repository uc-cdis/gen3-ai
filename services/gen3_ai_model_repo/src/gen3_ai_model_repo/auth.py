from fastapi import HTTPException

from gen3_ai_model_repo.config import MODEL_REPO_TOKEN


def validate_token(authorization: str | None):
    """Validate the incoming authorization header for the mock API."""
    if authorization != MODEL_REPO_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
