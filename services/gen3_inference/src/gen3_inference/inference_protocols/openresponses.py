from collections.abc import AsyncGenerator

from fastapi.responses import JSONResponse, StreamingResponse
from openai import OpenAI
from openresponses_types import (
    CreateResponseBody,
    Object,
    ResponseCompletedStreamingEvent,
    ResponseCreatedStreamingEvent,
    ResponseInProgressStreamingEvent,
    ResponseResource,
    Type37,
    Type39,
    Type40,
)

from gen3_inference.inference_protocols.base import InferenceProtocolClient
from gen3_inference.inference_protocols.utils.openai_to_openresponses import openai_response_to_openresponses

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


class OpenResponsesClient(InferenceProtocolClient):
    NAME = "openresponses"

    def __init__(self):
        super().__init__()

    async def generate_streaming_response(self, body: CreateResponseBody):
        return self._create_streaming_response(body)

    async def generate_non_streaming_response(self, body: CreateResponseBody):
        return self._create_non_streaming_response(body)

    def _create_non_streaming_response(self, body):
        client = OpenAI(
            # TODO: FIXME: actually add thes from the model info
            # For local testing: ollama supports Responses
            # https://docs.ollama.com/api/openai-compatibility#simple-/v1/responses-example
            api_key="ollama",
            base_url="http://localhost:11434/v1/",
            organization="Gen3",
            project="Gen3",
            webhook_secret="ollama",
        )

        response = client.responses.create(
            # TODO: probably need to pass in a ton more stuff from the request
            model=body.model,
            instructions=body.instructions,
            input=body.input,
            stream=False,
        )
        standard_response = openai_response_to_openresponses(
            response, metadata={"inference_protocol_client": self.NAME}
        )

        return JSONResponse(content=standard_response.model_dump())

    # TODO: FIXME: Actually implement streaming response
    def _create_streaming_response(self, body):
        async def event_generator() -> AsyncGenerator[str]:
            seq = 0

            # build a base ResponseResource snapshot to reuse in events
            base_response = ResponseResource(
                id="resp_stream_dummy_id",
                object=Object.response,
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
                metadata=(body.metadata or {}).update({"inference_protocol_client": self.NAME}),
                safety_identifier=body.safety_identifier,
                prompt_cache_key=body.prompt_cache_key,
            )

            # 1. response.created
            created_event = ResponseCreatedStreamingEvent(
                type=Type37("response.created"),
                sequence_number=seq,
                response=base_response,
            )
            yield f"data: {created_event.model_dump_json()}\n\n"
            seq += 1

            # 2. response.in_progress (stub)
            in_progress_event = ResponseInProgressStreamingEvent(
                type=Type39("response.in_progress"),
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
            completed_response = base_response.model_copy(update={"status": "completed", "completed_at": 0})
            completed_event = ResponseCompletedStreamingEvent(
                type=Type40("response.completed"),
                sequence_number=seq,
                response=completed_response,
            )
            yield f"data: {completed_event.model_dump_json()}\n\n"
            seq += 1

            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")
