"""Response helpers for the Gen3 AI model repo service."""

from fastapi.responses import RedirectResponse


def build_head_response(commit_hash: str, etag: str, size: int, signed_url: str) -> RedirectResponse:
    """
    Build a HEAD-style redirect response for repository file access.

    Returns:
        RedirectResponse: A redirect response with custom headers for file metadata.
    """
    headers = {
        "X-Repo-Commit": commit_hash,
        "X-Linked-Etag": etag,
        "X-Linked-Size": str(size),
        "Location": signed_url,
    }
    return RedirectResponse(url=signed_url, headers=headers, status_code=302)
