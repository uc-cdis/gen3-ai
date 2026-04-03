import json
from unittest.mock import AsyncMock, MagicMock

from fastapi.responses import StreamingResponse
from openresponses_types import (
    CreateResponseBody,
    FunctionToolChoice,
    InputTextContent,
    InputTextContentParam,
    Message,
    MessageRole,
    MessageStatus,
    Object,
    ResponseResource,
    ResponseCreatedStreamingEvent,
    Role,
    TextField,
    TextResponseFormat,
    TruncationEnum,
    Type31,
    Type34,
    Type37,
    UserMessageItemParam,
    VerbosityEnum,
)
import pytest
from fastapi.testclient import TestClient

from gen3_inference.main import app_instance

TEST_AI_MODEL = "test-ai-model"

MOCK_RESPONSE = ResponseResource(
    id="resp_001",
    object=Object.response,
    created_at=1634567890,
    completed_at=None,
    status="in_progress",
    incomplete_details=None,
    model=TEST_AI_MODEL,
    previous_response_id=None,
    instructions="Write a short story",
    output=[
        Message(
            type="message",
            id="foobar",
            status=MessageStatus.completed,
            role=MessageRole.assistant,
            content=list([InputTextContent(type="input_text", text="Hello world")]),
        ),
    ],
    error=None,
    tools=[],
    tool_choice=FunctionToolChoice(type=Type31(value="function")),
    truncation=TruncationEnum(value="auto"),
    parallel_tool_calls=False,
    text=TextField(format=TextResponseFormat(type=Type34(value="text")), verbosity=VerbosityEnum(value="medium")),
    top_p=1.0,
    presence_penalty=0.0,
    frequency_penalty=0.0,
    top_logprobs=0,
    temperature=1.0,
    reasoning=None,
    usage=None,
    max_output_tokens=None,
    max_tool_calls=None,
    store=False,
    background=False,
    service_tier="default",
    metadata={"inference_protocol_client": "openresponses"},
    safety_identifier=None,
    prompt_cache_key=None,
)

VALID_NON_STREAMED_RESPONSE = json.loads(MOCK_RESPONSE.model_dump_json())


def _single_created_sse_event() -> str:
    """
    Return a single streaming event that mimics the real implementation.
    """
    event = ResponseCreatedStreamingEvent(
        type=Type37(value="response.created"),
        sequence_number=0,
        response=MOCK_RESPONSE,
    )
    return f"data: {event.model_dump_json()}\n\n"


VALID_STREAMED_RESPONSE = _single_created_sse_event() + "data: [DONE]\n\n"

VALID_AI_MODEL_INFO = {
    "url": "https://example.com",
    "metadata": {"name": TEST_AI_MODEL, "inference_protocol_clients": ["openresponses"]},
}


@pytest.fixture
def mock_inference_client():
    inference_client = MagicMock()
    inference_client.NAME = "mock_inference_client"
    inference_client.generate_non_streaming_response = AsyncMock(return_value=VALID_NON_STREAMED_RESPONSE)

    # setup streaming
    async def stream_generator():
        # fastapi will iterate over this generator and write the
        # yielded string directly to the HTTP body
        yield VALID_STREAMED_RESPONSE

    inference_client.generate_streaming_response = MagicMock(
        return_value=StreamingResponse(stream_generator(), media_type="text/event-stream")
    )

    return inference_client


@pytest.fixture
def valid_user_msg_body_non_streaming() -> CreateResponseBody:
    return CreateResponseBody(
        model=TEST_AI_MODEL,
        input=[
            UserMessageItemParam(
                type="message",
                role=Role("user"),
                content=[InputTextContentParam(type="input_text", text="Hello, world!")],
                status="completed",
            )
        ],
        stream=False,
    )


@pytest.fixture(scope="session")
def client() -> TestClient:
    """
    A TestClient that wraps the app
    for ALL tests and ALL workers
    """
    return TestClient(app_instance)
