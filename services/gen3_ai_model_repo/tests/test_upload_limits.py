"""Regression tests for upload request limits before multipart parsing."""

from fastapi.testclient import TestClient

from gen3_ai_model_repo.main import get_app


def _client(monkeypatch, limit: int) -> TestClient:
    """Build an app with a small wire limit and a storage sentinel."""
    import gen3_ai_model_repo.main as main

    monkeypatch.setattr(main.config, "MAX_UPLOAD_BYTES", limit)
    monkeypatch.setattr(main, "get_storage_provider", lambda: (_ for _ in ()).throw(AssertionError("storage called")))
    return TestClient(get_app())


def test_declared_oversize_is_rejected_before_multipart_parsing(monkeypatch):
    """A declared oversized multipart body returns 413 without invoking storage."""
    client = _client(monkeypatch, 1)
    response = client.post(
        "/api/models/ns/repo/upload",
        content=b"not parsed",
        headers={"content-type": "multipart/form-data; boundary=test", "content-length": str(2 * 1024 * 1024)},
    )
    assert response.status_code == 413


def test_streamed_oversize_without_content_length_returns_413(monkeypatch):
    """A chunked body crossing the wire limit becomes a 413 response."""
    client = _client(monkeypatch, 1)

    def chunks():
        yield b'--test\r\nContent-Disposition: form-data; name="files"; filename="x"\r\n\r\n'
        yield b"x" * (1024 * 1024)
        yield b"\r\n--test--\r\n"

    response = client.post(
        "/api/models/ns/repo/upload",
        content=chunks(),
        headers={"content-type": "multipart/form-data; boundary=test"},
    )
    assert response.status_code == 413


def test_body_at_wire_boundary_is_not_rejected_by_middleware(monkeypatch):
    """A body at the middleware boundary proceeds to normal multipart handling."""
    client = _client(monkeypatch, 0)
    response = client.post(
        "/api/models/ns/repo/upload",
        content=b"x" * (1024 * 1024),
        headers={"content-type": "multipart/form-data; boundary=test"},
    )
    assert response.status_code != 413
