import json
from gen3_inference.types import OpenResponsesError
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from fastapi.responses import StreamingResponse
from httpx import Response
from unittest.mock import AsyncMock, MagicMock, patch

from openresponses_types.types import CreateResponseBody
from gen3_inference.errors import (
    ERROR_TYPE_INVALID_REQUEST,
    ERROR_TYPE_NOT_FOUND,
)

from conftest import TEST_AI_MODEL, VALID_NON_STREAMED_RESPONSE, VALID_STREAMED_RESPONSE, VALID_AI_MODEL_INFO


@pytest.mark.asyncio
@patch("gen3_inference.routes.responses.get_ai_model_info")
async def test_create_response_non_streaming_positive(
    mock_get_model: AsyncMock,
    valid_user_msg_body_non_streaming: CreateResponseBody,
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
):
    """
    Verify that a normal (JSON) response works and that the JSON
    response is validated against ResponseResource.
    """
    mock_get_model.return_value = VALID_AI_MODEL_INFO

    # Fake client that returns a dict that satisfies ResponseResource
    fake_client = MagicMock()

    mocked_return = VALID_NON_STREAMED_RESPONSE
    mocked_return.update({"id": "foo", "status": "completed"})

    fake_client.generate_non_streaming_response = AsyncMock(return_value=mocked_return)
    monkeypatch.setattr(
        "gen3_inference.routes.responses.get_inference_protocol_client",
        AsyncMock(return_value=fake_client),
    )

    resp = client.post("/responses", json=json.loads(valid_user_msg_body_non_streaming.model_dump_json()))
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("id") == "foo"
    assert data.get("status") == "completed"


@pytest.mark.asyncio
@patch("gen3_inference.routes.responses.get_ai_model_info")
async def test_create_response_streaming_positive(
    mock_get_model: AsyncMock,
    valid_user_msg_body_non_streaming: CreateResponseBody,
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
):
    """
    Verify that the streaming branch returns a StreamingResponse
    """
    mock_get_model.return_value = VALID_AI_MODEL_INFO

    mocked_return = VALID_STREAMED_RESPONSE
    mocked_return = mocked_return.replace("resp_001", "test_response_id_456")

    # Fake client that returns a real StreamingResponse
    async def _generator():
        yield mocked_return

    stream_resp = StreamingResponse(_generator(), media_type="text/event-stream")

    fake_client = MagicMock()
    fake_client.generate_streaming_response = MagicMock(return_value=stream_resp)
    monkeypatch.setattr(
        "gen3_inference.routes.responses.get_inference_protocol_client",
        AsyncMock(return_value=fake_client),
    )

    body = valid_user_msg_body_non_streaming.model_copy(update={"stream": True})
    resp = client.post("/responses", json=json.loads(body.model_dump_json()))

    assert resp.status_code == 200
    # response should be: text/event-stream; charset=utf-8
    # assuming that charset=utf-8 is okay for compliance with the spec
    assert "text/event-stream" in resp.headers["content-type"]
    content = resp.content.decode()
    assert '"test_response_id_456"' in content


@pytest.mark.asyncio
async def test_get_ai_model_info_missing_model(
    valid_user_msg_body_non_streaming: CreateResponseBody, monkeypatch: pytest.MonkeyPatch, client: TestClient
):
    """
    Body has no model - 400.
    """
    body = valid_user_msg_body_non_streaming.model_copy(update={"model": ""})

    resp = client.post("/responses", json=json.loads(body.model_dump_json()))

    assert resp.status_code == 400
    data = resp.json()
    assert data.get("error", {}).get("type") == ERROR_TYPE_INVALID_REQUEST


@pytest.mark.asyncio
async def test_get_ai_model_info_not_found(
    valid_user_msg_body_non_streaming: CreateResponseBody, monkeypatch: pytest.MonkeyPatch, client: TestClient
):
    """
    Primary host returns 404 and all trusted hosts return 404 - 404.
    """
    body = valid_user_msg_body_non_streaming.model_copy()

    # Mock httpx.AsyncClient so that every GET returns 404
    class DummyClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        async def get(self, url):
            return Response(status_code=404)

    monkeypatch.setattr("httpx.AsyncClient", lambda *a, **k: DummyClient())

    # Mock config to give one trusted host
    monkeypatch.setattr("gen3_inference.config.ALLOWED_GEN3_INFERENCE_HOSTS", {"https://trusted.com"})

    resp = client.post("/responses", json=json.loads(body.model_dump_json()))

    assert resp.status_code == 404
    data = resp.json()
    assert data.get("error", {}).get("type") == ERROR_TYPE_NOT_FOUND


@pytest.mark.asyncio
@patch("gen3_inference.routes.responses.get_inference_protocol_client")
async def test_get_ai_model_info_trusted_domain_success(
    mock_get_client: AsyncMock,
    mock_inference_client: MagicMock,
    valid_user_msg_body_non_streaming: CreateResponseBody,
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
):
    """
    Primary 404 but a trusted domain returns the model - success.
    """
    test_inference_protocol_client = "test_protocol_client"
    mocked_response = VALID_NON_STREAMED_RESPONSE
    mocked_response.update({"metadata": {"inference_protocol_client": test_inference_protocol_client}})
    mock_inference_client.return_value.generate_non_streaming_response.return_value = AsyncMock(
        return_value=mocked_response
    )

    mock_get_client.return_value = mock_inference_client

    body = valid_user_msg_body_non_streaming.model_copy()

    class DummyClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        async def get(self, url):
            if url.startswith("https://trusted.com"):
                return Response(
                    status_code=200,
                    json={
                        "ai_model_info": {
                            "name": TEST_AI_MODEL,
                            "inference_protocol_clients": [test_inference_protocol_client],
                        },
                    },
                )
            return Response(status_code=404)

    monkeypatch.setattr("httpx.AsyncClient", lambda *a, **k: DummyClient())
    monkeypatch.setattr("gen3_inference.config.ALLOWED_GEN3_INFERENCE_HOSTS", {"https://trusted.com"})
    monkeypatch.setattr("gen3_inference.config.GEN3_AI_MODEL_REPO_URL", "https://primary.com")

    resp = client.post("/responses", json=json.loads(body.model_dump_json()))

    assert resp.status_code == 200
    data = resp.json()
    assert not data.get("error", {})
    assert data.get("model") == TEST_AI_MODEL
    assert data.get("metadata", {}).get("inference_protocol_client") == test_inference_protocol_client


@pytest.mark.asyncio
@patch("gen3_inference.routes.responses.get_ai_model_info")
async def test_get_inference_protocol_client_unknown(
    mock_get_model: AsyncMock,
    valid_user_msg_body_non_streaming: CreateResponseBody,
    client: TestClient,
):
    """
    Model found but advertises an unsupported protocol - 400.
    """
    mocked_return = VALID_AI_MODEL_INFO
    mocked_return.update({"inference_protocol_clients": ["unsupported"]})
    mock_get_model.return_value = mocked_return

    body = valid_user_msg_body_non_streaming.model_copy()
    resp = client.post("/responses", json=json.loads(body.model_dump_json()))

    assert resp.status_code == 400
    data = resp.json()
    assert data.get("error", {}).get("type") == ERROR_TYPE_INVALID_REQUEST


@pytest.mark.asyncio
@patch("gen3_inference.routes.responses.get_ai_model_info")
async def test_get_inference_protocol_client_empty(
    mock_get_model: AsyncMock,
    valid_user_msg_body_non_streaming: CreateResponseBody,
    client: TestClient,
):
    """Empty list of protocol names - 400."""
    mocked_return = VALID_AI_MODEL_INFO
    # empty
    mocked_return.update({"inference_protocol_clients": []})
    mock_get_model.return_value = mocked_return

    body = valid_user_msg_body_non_streaming.model_copy()
    resp = client.post("/responses", json=json.loads(body.model_dump_json()))

    assert resp.status_code == 400
    data = resp.json()
    assert data.get("error", {}).get("type") == ERROR_TYPE_INVALID_REQUEST


@pytest.mark.asyncio
async def test_create_response_missing_model(valid_user_msg_body_non_streaming: CreateResponseBody, client: TestClient):
    """
    Send a request where the body contains no model field.
    The endpoint should return 400 before reaching the protocol client.
    """
    body = valid_user_msg_body_non_streaming.model_copy(update={"model": ""})
    resp = client.post("/responses", json=json.loads(body.model_dump_json()))
    assert resp.status_code == 400
    assert resp.json().get("error", {}).get("code") == ERROR_TYPE_INVALID_REQUEST


@pytest.mark.asyncio
async def test_create_response_unknown_protocol(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, valid_user_msg_body_non_streaming: CreateResponseBody
):
    """
    The model advertises an inference protocol that the service doesn't
    understand. Expect 400.
    """

    # Fake the get_ai_model_info to return a model with a bogus protocol
    async def fake_ai_info(_body):
        return {"inference_protocol_clients": ["unknown"]}

    monkeypatch.setattr("gen3_inference.routes.responses.get_ai_model_info", fake_ai_info)

    resp = client.post("/responses", json=json.loads(valid_user_msg_body_non_streaming.model_dump_json()))
    assert resp.status_code == 400
    assert resp.json().get("error", {}).get("code") == ERROR_TYPE_INVALID_REQUEST


@pytest.mark.asyncio
@patch("gen3_inference.routes.responses.get_ai_model_info")
async def test_create_response_model_not_found(
    mock_get_model: AsyncMock,
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
    valid_user_msg_body_non_streaming: CreateResponseBody,
):
    """
    The model is requested, but the repository (primary + trusted) does not
    contain it. Endpoint should propagate the 404 from get_ai_model_info.
    """
    mock_get_model.side_effect = HTTPException(
        status_code=404,
        detail=OpenResponsesError(type=ERROR_TYPE_NOT_FOUND, code=ERROR_TYPE_NOT_FOUND, message="not found").to_json(),
    )

    resp = client.post("/responses", json=json.loads(valid_user_msg_body_non_streaming.model_dump_json()))
    assert resp.status_code == 404
    assert resp.json().get("error", {}).get("code") == ERROR_TYPE_NOT_FOUND


@pytest.mark.asyncio
@patch("gen3_inference.routes.responses.get_ai_model_info")
async def test_create_response_bogus_protocol(
    mock_get_model: AsyncMock,
    valid_user_msg_body_non_streaming: CreateResponseBody,
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
):
    """
    Simulate the normal path but let get_inference_protocol_client raise.
    The endpoint should return a 400 error.
    """
    # Fake get_ai_model_info to return a model that advertises only
    # an unsupported protocol
    VALID_AI_MODEL_INFO.update({"inference_protocol_clients": ["unknown"]})
    mock_get_model.return_value = VALID_AI_MODEL_INFO

    resp = client.post("/responses", json=json.loads(valid_user_msg_body_non_streaming.model_dump_json()))
    assert resp.status_code == 400
    assert resp.json().get("error", {}).get("code") == ERROR_TYPE_INVALID_REQUEST


@pytest.mark.asyncio
@patch("gen3_inference.routes.responses.get_ai_model_info")
async def test_create_response_with_openresponses_client(
    mock_get_model: AsyncMock,
    valid_user_msg_body_non_streaming: CreateResponseBody,
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
):
    """
    Verify that the router actually chooses OpenResponsesClient when it is
    advertised in the model's inference_protocol_clients list.
    """
    mock_get_model.return_value = VALID_AI_MODEL_INFO

    mocked_return = VALID_NON_STREAMED_RESPONSE
    mocked_return.update({"id": "test_create_response_with_openresponses_client"})

    # Patch the helper to return a fake OpenResponsesClient
    fake_client = MagicMock()
    fake_client.generate_non_streaming_response = AsyncMock(return_value=VALID_NON_STREAMED_RESPONSE)
    fake_client.NAME = "openresponses"

    async def fake_get_client(_names):
        return fake_client

    monkeypatch.setattr("gen3_inference.routes.responses.get_inference_protocol_client", fake_get_client)

    # Now hit the endpoint
    resp = client.post("/responses", json=json.loads(valid_user_msg_body_non_streaming.model_dump_json()))
    assert resp.status_code == 200
    assert resp.json()["id"] == "test_create_response_with_openresponses_client"
    assert fake_client.generate_non_streaming_response.called
