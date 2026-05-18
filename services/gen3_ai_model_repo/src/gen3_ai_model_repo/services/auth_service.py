from fastapi import HTTPException


class AuthService:
    def __init__(self):
        pass

    def validate_token(self, authorization: str | None):
        if authorization != "Bearer mock-token-123456":
            raise HTTPException(status_code=401, detail="Unauthorized")
