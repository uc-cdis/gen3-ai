from collections.abc import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from openresponses_types.types import (
    # Request body
    CreateResponseBody,
    ErrorStreamingEvent,
    ResponseCompletedStreamingEvent,
    ResponseContentPartAddedStreamingEvent,
    ResponseContentPartDoneStreamingEvent,
    # Streaming events
    ResponseCreatedStreamingEvent,
    ResponseFailedStreamingEvent,
    ResponseFunctionCallArgumentsDeltaStreamingEvent,
    ResponseFunctionCallArgumentsDoneStreamingEvent,
    ResponseIncompleteStreamingEvent,
    ResponseInProgressStreamingEvent,
    ResponseOutputItemAddedStreamingEvent,
    ResponseOutputItemDoneStreamingEvent,
    ResponseOutputTextAnnotationAddedStreamingEvent,
    ResponseOutputTextDeltaStreamingEvent,
    ResponseOutputTextDoneStreamingEvent,
    ResponseQueuedStreamingEvent,
    ResponseReasoningDeltaStreamingEvent,
    ResponseReasoningDoneStreamingEvent,
    ResponseReasoningSummaryDeltaStreamingEvent,
    ResponseReasoningSummaryDoneStreamingEvent,
    ResponseReasoningSummaryPartAddedStreamingEvent,
    ResponseReasoningSummaryPartDoneStreamingEvent,
    ResponseRefusalDeltaStreamingEvent,
    ResponseRefusalDoneStreamingEvent,
    # Core response resource
    ResponseResource,
)

responses_router = APIRouter()

MOCK_OUTPUT = [
    # ItemFields
    # Message
    {
        "type": "message",
        "id": "",
        "status": "completed",
        "role": "assistant",
        "content": [
            {
                # InputTextContent
                "type": "input_text",
                "text": "foobar",
            }
        ],
    },
    # Function Call
    {
        "type": "function_call",
        "id": "0",
        "call_id": "0",
        "name": "test",
        "arguments": "foo=bar",
        "status": "completed",
        "output": [
            {
                # InputTextContent
                "type": "input_text",
                "text": "foobar",
            }
        ],
    },
]


@responses_router.post(
    "/responses",
    response_model=ResponseResource,
    responses={
        200: {
            "content": {
                "application/json": {"schema": ResponseResource.model_json_schema()},
                "text/event-stream": {
                    "schema": {
                        "oneOf": [
                            ResponseCreatedStreamingEvent.model_json_schema(),
                            ResponseQueuedStreamingEvent.model_json_schema(),
                            ResponseInProgressStreamingEvent.model_json_schema(),
                            ResponseCompletedStreamingEvent.model_json_schema(),
                            ResponseFailedStreamingEvent.model_json_schema(),
                            ResponseIncompleteStreamingEvent.model_json_schema(),
                            ResponseOutputItemAddedStreamingEvent.model_json_schema(),
                            ResponseOutputItemDoneStreamingEvent.model_json_schema(),
                            ResponseReasoningSummaryPartAddedStreamingEvent.model_json_schema(),
                            ResponseReasoningSummaryPartDoneStreamingEvent.model_json_schema(),
                            ResponseContentPartAddedStreamingEvent.model_json_schema(),
                            ResponseContentPartDoneStreamingEvent.model_json_schema(),
                            ResponseOutputTextDeltaStreamingEvent.model_json_schema(),
                            ResponseOutputTextDoneStreamingEvent.model_json_schema(),
                            ResponseRefusalDeltaStreamingEvent.model_json_schema(),
                            ResponseRefusalDoneStreamingEvent.model_json_schema(),
                            ResponseReasoningDeltaStreamingEvent.model_json_schema(),
                            ResponseReasoningDoneStreamingEvent.model_json_schema(),
                            ResponseReasoningSummaryDeltaStreamingEvent.model_json_schema(),
                            ResponseReasoningSummaryDoneStreamingEvent.model_json_schema(),
                            ResponseOutputTextAnnotationAddedStreamingEvent.model_json_schema(),
                            ResponseFunctionCallArgumentsDeltaStreamingEvent.model_json_schema(),
                            ResponseFunctionCallArgumentsDoneStreamingEvent.model_json_schema(),
                            ErrorStreamingEvent.model_json_schema(),
                        ]
                    }
                },
            }
        }
    },
    tags=["Inference"],
)
async def create_response(
    body: CreateResponseBody,
):
    """
    Implements the /responses endpoint defined in the Open Responses OpenAPI spec,
    using the models from openresponses_types.types
    """
    if not body.stream:
        # ResponseResource as JSON
        return _create_non_streaming_response(body)
    else:
        # text/event-stream of streaming events
        return _create_streaming_response(body)


def _create_non_streaming_response(body):
    # TODO: actually implement

    # Here we build a minimal valid ResponseResource using incoming config.
    response = ResponseResource(
        id="resp_dummy_id",
        object="response",
        created_at=0,
        completed_at=0,
        status="completed",
        incomplete_details=None,
        model=body.model or "dummy-model",
        previous_response_id=body.previous_response_id,
        instructions=body.instructions,
        output=MOCK_OUTPUT,
        error=None,
        tools=[],
        tool_choice=body.tool_choice or "auto",
        truncation=body.truncation or "auto",
        parallel_tool_calls=bool(body.parallel_tool_calls),
        text={"format": {"type": "text"}, "verbosity": "medium"},
        top_p=body.top_p or 1.0,
        presence_penalty=body.presence_penalty or 0.0,
        frequency_penalty=body.frequency_penalty or 0.0,
        top_logprobs=body.top_logprobs or 0,
        temperature=body.temperature or 1.0,
        reasoning=body.reasoning,
        usage=None,
        max_output_tokens=body.max_output_tokens,
        max_tool_calls=body.max_tool_calls,
        store=bool(body.store),
        background=bool(body.background),
        service_tier=body.service_tier or "default",
        metadata=body.metadata or {},
        safety_identifier=body.safety_identifier,
        prompt_cache_key=body.prompt_cache_key,
    )
    return JSONResponse(content=response.model_dump())


def _create_streaming_response(body):
    async def event_generator() -> AsyncGenerator[str]:
        seq = 0

        # build a base ResponseResource snapshot to reuse in events
        base_response = ResponseResource(
            id="resp_stream_dummy_id",
            object="response",
            created_at=0,
            completed_at=None,
            status="in_progress",
            incomplete_details=None,
            model=body.model or "dummy-model",
            previous_response_id=body.previous_response_id,
            instructions=body.instructions,
            output=MOCK_OUTPUT,
            error=None,
            tools=[],
            tool_choice=body.tool_choice or "auto",
            truncation=body.truncation or "auto",
            parallel_tool_calls=bool(body.parallel_tool_calls),
            text={"format": {"type": "text"}, "verbosity": "medium"},
            top_p=body.top_p or 1.0,
            presence_penalty=body.presence_penalty or 0.0,
            frequency_penalty=body.frequency_penalty or 0.0,
            top_logprobs=body.top_logprobs or 0,
            temperature=body.temperature or 1.0,
            reasoning=body.reasoning,
            usage=None,
            max_output_tokens=body.max_output_tokens,
            max_tool_calls=body.max_tool_calls,
            store=bool(body.store),
            background=bool(body.background),
            service_tier=body.service_tier or "default",
            metadata=body.metadata or {},
            safety_identifier=body.safety_identifier,
            prompt_cache_key=body.prompt_cache_key,
        )

        # 1. response.created
        created_event = ResponseCreatedStreamingEvent(
            type="response.created",
            sequence_number=seq,
            response=base_response,
        )
        yield f"data: {created_event.model_dump_json()}\n\n"
        seq += 1

        # 2. response.in_progress (stub)
        in_progress_event = ResponseInProgressStreamingEvent(
            type="response.in_progress",
            sequence_number=seq,
            response=base_response,
        )
        yield f"data: {in_progress_event.model_dump_json()}\n\n"
        seq += 1

        # TODO: in real implementation, between in_progress and completed you
        # would:
        #   - stream ResponseOutputTextDeltaStreamingEvent / ...Done events
        #   - stream tool call events
        #   - update response snapshot as generation proceeds

        # 3. response.completed (stub)
        completed_response = base_response.copy(update={"status": "completed", "completed_at": 0})
        completed_event = ResponseCompletedStreamingEvent(
            type="response.completed",
            sequence_number=seq,
            response=completed_response,
        )
        yield f"data: {completed_event.model_dump_json()}\n\n"
        seq += 1

        # TODO: is this required?
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
