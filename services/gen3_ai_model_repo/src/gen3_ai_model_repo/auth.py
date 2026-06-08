from fastapi import Header, HTTPException

from gen3_ai_model_repo.config import MODEL_REPO_TOKEN


def validate_token(authorization: str | None):
    """
    Validate the incoming authorization header for the model repository API.

    WARNING: Security Considerations
    - The token is compared as a simple string match. For production use, consider:
      - Using JWT tokens for more secure authentication
      - Implementing token expiration and refresh mechanisms
      - Using HTTPS/TLS for all API communications
      - Implementing rate limiting and other security measures

    Args:
        authorization: The authorization header value

    Raises:
        HTTPException: 401 if authorization header is missing or invalid
    """
    if authorization != MODEL_REPO_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")


async def verify_authorization(authorization: str | None = Header(default=None)):
    """
    FastAPI dependency for authentication.

    This function can be used with FastAPI's Depends to automatically validate
    the authorization header on protected routes.

    Args:
        authorization: The authorization header value (automatically extracted by FastAPI)

    Raises:
        HTTPException: 401 if authorization is invalid
    """
    validate_token(authorization)
