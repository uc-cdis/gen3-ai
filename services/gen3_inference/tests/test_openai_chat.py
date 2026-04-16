"""
Tests for OpenAI Chat Completions utils
"""

import json

import pytest
from unittest.mock import MagicMock, patch
from gen3_inference.inference_protocols.openai_chat import OpenaiChat

from gen3_inference.inference_protocols.utils.openai_chat_to_openresponses import (
    chat_completion_to_openresponses_response,
)


@pytest.mark.asyncio
@patch("gen3_inference.inference_protocols.openai_chat.OpenAI")
async def test_generate_non_streaming_response_calls_client(
    mock_openai_class: MagicMock,
):
    """
    Test that generate_non_streaming_response calls OpenAI client with correct parameters
    """
    # mock response
    mock_response = MagicMock()
    mock_response.id = "chatcmpl-789"
    mock_response.object = "response"
    mock_response.created = 1677652290
    mock_response.model = "test-model"
    mock_response.service_tier = "default"
    mock_response.choices = [
        MagicMock(index=0, message=MagicMock(role="assistant", content="Success"), finish_reason="stop")
    ]
    mock_response.usage = MagicMock(prompt_tokens=5, completion_tokens=7, total_tokens=12)

    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_client.chat.completions.create.return_value = mock_response

    # mock request body
    request_body = MagicMock()
    request_body.model = "test-model"

    msg = MagicMock()
    msg.type = "message"

    role = MagicMock()
    role.value = "user"
    msg.role = role

    content = MagicMock()
    content.type = "input_text"
    content.text = "Hello, world!"
    msg.content = [content]

    request_body.input = [msg]

    request_body.stream_options = None

    protocol = OpenaiChat(base_url="http://localhost:11434/v1")

    response = await protocol.generate_non_streaming_response(request_body, {"auth_token": "test-key"})

    assert mock_client.chat.completions.create.called
    assert mock_client.chat.completions.create.call_args[1]["model"] == "test-model"
    assert mock_client.chat.completions.create.call_args[1]["stream"] is False

    response_data = json.loads(response.body)
    assert response_data["id"] == "chatcmpl-789"
    assert response_data["model"] == "test-model"
    assert response_data["output"][0]["content"][0]["text"] == "Success"
    assert response_data["usage"]["total_tokens"] == 12


@pytest.mark.asyncio
@patch("gen3_inference.inference_protocols.openai_chat.OpenAI")
async def test_generate_streaming_response_calls_client(
    mock_openai_class: MagicMock,
):
    """
    Test that generate_streaming_response calls OpenAI client with stream=True
    """
    # mock chunks
    chunks = [
        MagicMock(
            model_dump=lambda: {
                "id": "chatcmpl-101",
                "object": "chat.completion.chunk",
                "created": 1677652291,
                "model": "test-model",
                "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}],
            }
        ),
        MagicMock(
            model_dump=lambda: {
                "id": "chatcmpl-101",
                "object": "chat.completion.chunk",
                "created": 1677652291,
                "model": "test-model",
                "choices": [{"index": 0, "delta": {"content": " world"}, "finish_reason": "stop"}],
                "finish_reason": "stop",
            },
            usage=MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        ),
    ]

    mock_stream = MagicMock()
    mock_stream.__iter__ = MagicMock(return_value=iter(chunks))

    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_client.chat.completions.create.return_value = mock_stream

    # mock request body
    request_body = MagicMock()
    request_body.model = "test-model"

    msg = MagicMock()
    msg.type = "message"

    role = MagicMock()
    role.value = "user"
    msg.role = role

    content = MagicMock()
    content.type = "input_text"
    content.text = "Hello, world!"
    msg.content = [content]

    request_body.input = [msg]

    request_body.stream_options = None

    protocol = OpenaiChat(base_url="http://localhost:11434/v1")

    response = protocol.generate_streaming_response(request_body, {"auth_token": "test-key"})

    assert mock_client.chat.completions.create.called
    assert mock_client.chat.completions.create.call_args[1]["model"] == "test-model"
    assert mock_client.chat.completions.create.call_args[1]["stream"] is True

    captured_events = []
    async for chunk in response.body_iterator:
        lines = chunk.decode("utf-8").split("\n")
        for line in lines:
            if line.startswith("data: ") and line != "data: [DONE]":
                # remove 'event: ...\n' prefix to parse the data
                data_str = line[6:]
                captured_events.append(json.loads(data_str))

    deltas = [e["delta"] for e in captured_events if e.get("type") == "response.output_text.delta"]
    assert "Hello" in deltas
    assert " world" in deltas

    # check usage was present in the final chunk
    usage_event = next((e for e in captured_events if e.get("type") == "response.output_text.done"), None)
    assert usage_event is not None
    assert usage_event["usage"]["total_tokens"] == 2


def test_chat_completion_to_openresponses_response_basic():
    """
    Test basic non-streaming conversion returns correct structure
    """
    # mock response
    mock_response = MagicMock()
    mock_response.id = "chatcmpl-123"
    mock_response.object = "response"
    mock_response.created = 1677652288
    mock_response.model = "foobar"
    mock_response.service_tier = "default"
    mock_response.choices = [
        MagicMock(index=0, message=MagicMock(role="assistant", content="Hello"), finish_reason="stop")
    ]
    mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)

    response = chat_completion_to_openresponses_response(mock_response)

    assert response.id == "chatcmpl-123"
    assert response.object == "response"
    assert response.model == "foobar"
    assert response.output[0].content[0].text == "Hello"
    assert response.usage.total_tokens == 15


def test_chat_completion_to_openresponses_response_no_usage():
    """
    Test conversion without usage information
    """

    # mock response without usage
    mock_response = MagicMock()
    mock_response.id = "chatcmpl-456"
    mock_response.object = "response"
    mock_response.created = 1677652289
    mock_response.model = "foobar"
    mock_response.service_tier = "default"
    mock_response.choices = [
        MagicMock(index=0, message=MagicMock(role="assistant", content="Hi"), finish_reason="stop")
    ]
    mock_response.usage = False

    response = chat_completion_to_openresponses_response(mock_response)

    assert response.id == "chatcmpl-456"
    assert not response.usage
