import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from conftest import VALID_AI_MODEL_INFO, VALID_NON_STREAMED_RESPONSE, VALID_STREAMED_RESPONSE
from fastapi.testclient import TestClient
from openresponses_types import CreateResponseBody


@pytest.mark.asyncio
@patch("gen3_inference.routes.responses.get_ai_model_info")
@patch("gen3_inference.routes.responses.get_inference_protocol_client")
async def test_create_response_non_streaming_function_called(
    mock_get_client: AsyncMock,
    mock_get_model: AsyncMock,
    mock_inference_client: MagicMock,
    client: TestClient,
    valid_user_msg_body_non_streaming: CreateResponseBody,
):
    """
    Test the non-streaming path of `create_response`.

    Args:
        mock_get_client (AsyncMock): Patched protocol-client resolver.
        mock_get_model (AsyncMock): Patched model-info resolver.
        mock_inference_client (MagicMock): Mock inference protocol client.
        client (TestClient): FastAPI test client fixture.
        valid_user_msg_body_non_streaming (CreateResponseBody): Valid request body fixture.
    """
    mock_get_model.return_value = VALID_AI_MODEL_INFO
    mock_get_client.return_value = mock_inference_client

    input_body = json.loads(valid_user_msg_body_non_streaming.model_dump_json())
    response = client.post("/v1/responses", json=input_body)

    assert response.status_code == 200
    assert response.json() == VALID_NON_STREAMED_RESPONSE
    mock_get_model.assert_awaited()
    mock_get_client.assert_awaited()
    mock_inference_client.generate_non_streaming_response.assert_called()
    mock_inference_client.generate_streaming_response.assert_not_called()


@pytest.mark.asyncio
@patch("gen3_inference.routes.responses.get_ai_model_info")
@patch("gen3_inference.routes.responses.get_inference_protocol_client")
async def test_create_response_streaming_function_called(
    mock_get_client: AsyncMock,
    mock_get_model: AsyncMock,
    mock_inference_client: MagicMock,
    client: TestClient,
    valid_user_msg_body_non_streaming: CreateResponseBody,
):
    """
    Test the streaming path of `create_response`.

    Args:
        mock_get_client (AsyncMock): Patched protocol-client resolver.
        mock_get_model (AsyncMock): Patched model-info resolver.
        mock_inference_client (MagicMock): Mock inference protocol client.
        client (TestClient): FastAPI test client fixture.
        valid_user_msg_body_non_streaming (CreateResponseBody): Valid request body fixture.
    """
    mock_get_model.return_value = VALID_AI_MODEL_INFO
    mock_get_client.return_value = mock_inference_client

    input_body = json.loads(valid_user_msg_body_non_streaming.model_dump_json())
    input_body["stream"] = True

    response = client.post("/v1/responses", json=input_body)

    assert response.status_code == 200
    # the body of a StreamingResponse is a raw byte stream – decode it
    assert response.content.decode() == VALID_STREAMED_RESPONSE

    mock_get_model.assert_awaited()
    mock_get_client.assert_awaited()
    mock_inference_client.generate_streaming_response.assert_called()
    mock_inference_client.generate_non_streaming_response.assert_not_called()
