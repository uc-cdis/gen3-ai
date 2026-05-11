from fastapi.responses import RedirectResponse


class ResponseService:
    def __init__(self):
        pass

    def build_head_response(self, commit_hash: str, etag: str, size: int, signed_url: str) -> RedirectResponse:
        headers = {
            "X-Repo-Commit": commit_hash,
            "X-Linked-Etag": etag,
            "X-Linked-Size": str(size),
            "Location": signed_url,
        }
        return RedirectResponse(url=signed_url, headers=headers)
