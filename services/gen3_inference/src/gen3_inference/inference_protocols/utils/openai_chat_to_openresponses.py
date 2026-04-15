"""
Converter functions for converting between OpenAI Chat interface and Open Responses
"""

import json
from collections.abc import AsyncGenerator
from typing import Any

from fastapi.responses import StreamingResponse
from openai import Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessageFunctionToolCall
from openai.types.chat.chat_completion import Choice as OaiChatChoice
from openai.types.completion_usage import CompletionUsage as OaiCompletionUsage
from openresponses_types import (
    AllowedToolChoice,
    FunctionCall,
    FunctionCallOutput,
    FunctionTool,
    FunctionToolChoice,
    InputTextContent,
    InputTokensDetails,
    Message,
    MessageRole,
    MessageStatus,
    Object,
    OutputTokensDetails,
    ReasoningBody,
    ResponseResource,
    TextField,
    TextResponseFormat,
    ToolChoiceValueEnum,
    TruncationEnum,
    Type31,
    Type33,
    Type34,
    Usage,
    VerbosityEnum,
)

from gen3_inference.config import logging


def openresponses_request_to_chat_request(body: dict[str, Any]) -> dict[str, Any]:
    """
    Convert an Open Responses request body into a Chat Completions request

    Args:
        body (Dict[str, Any]): The Open Responses request body

    Returns:
        Dict[str, Any]: A dictionary suitable for the Chat Completions API
    """
    chat_body: dict[str, Any] = {}
    chat_body["model"] = body.get("model")

    message_role_map = {
        MessageRole.user: "user",
        MessageRole.assistant: "assistant",
        MessageRole.system: "system",
    }

    instructions = body.get("instructions")
    input_val = body.get("input")

    messages: list[dict[str, Any]] = []

    if instructions:
        messages.append({"role": "system", "content": instructions})

    if isinstance(input_val, list):
        for msg in input_val:
            role = msg.get("role")
            content = msg.get("content")
            role_normalized = message_role_map.get(role, "user")
            messages.append({"role": role_normalized, "content": content})

    elif isinstance(input_val, str):
        messages.append({"role": "user", "content": input_val})

    else:
        messages.append({"role": "user", "content": str(input_val)})

    chat_body["messages"] = messages

    temperature = body.get("temperature")
    if temperature is not None:
        chat_body["temperature"] = temperature

    top_p = body.get("top_p")
    if top_p is not None:
        chat_body["top_p"] = top_p

    max_output_tokens = body.get("max_output_tokens")
    if max_output_tokens is not None:
        chat_body["max_tokens"] = max_output_tokens

    tools = body.get("tools", [])
    if tools:
        chat_tools = []
        for tool in tools:
            if isinstance(tool, FunctionTool):
                chat_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.parameters,
                            "strict": tool.strict,
                        },
                    }
                )
            chat_body["tools"] = chat_tools

    tool_choice = body.get("tool_choice")
    if tool_choice is not None:
        if isinstance(tool_choice, ToolChoiceValueEnum):
            chat_body["tool_choice"] = tool_choice.value
        elif isinstance(tool_choice, FunctionToolChoice):
            chat_body["tool_choice"] = {"type": "function", "function": {"name": tool_choice.name}}
        elif isinstance(tool_choice, AllowedToolChoice):
            allowed = []
            for ft in tool_choice.tools:
                if isinstance(ft, FunctionTool):
                    allowed.append({"type": "function", "function": {"name": ft.name}})
            if allowed:
                chat_body["tool_choice"] = {"type": "function", "function": {"name": allowed[0]["function"]["name"]}}
        elif isinstance(tool_choice, dict):
            chat_body["tool_choice"] = tool_choice
        elif isinstance(tool_choice, str):
            chat_body["tool_choice"] = tool_choice

    return chat_body


def convert_chat_completion_stream_to_sse(
    chat_completion_stream: Stream[ChatCompletionChunk],
    include_usage: bool = True,
) -> StreamingResponse:
    """
    Convert OpenAI Chat Completions streaming chunks to Server-Sent Events (SSE) format
    following the Open Responses specification

    Open Responses mandates these events:
    1. response.output_item.added (with the message item)
    2. response.output_text.delta (for each chunk)
    3. response.output_text.done
    4. response.output_item.done

    Args:
        chat_completion_stream (Stream[ChatCompletionChunk]): The streaming chat completion
            response from OpenAI Chat interface
        include_usage (bool): Whether to include usage information in the final chunk

    Returns:
        StreamingResponse: FastAPI StreamingResponse with media_type "text/event-stream"
    """

    async def _generator() -> AsyncGenerator[bytes]:
        """
        Yield each chat completion chunk as SSE events following Open Responses spec
        """
        usage_info = None
        sequence_number = 0

        event_payload = {
            "type": "response.output_item.added",
            "sequence_number": sequence_number,
            "item": {
                "id": "first_event",
                "type": "message",
                "status": "in_progress",
                "content": [],
                "role": "assistant",
            },
        }
        yield b"event: response.output_item.added\n"
        yield f"data: {json.dumps(event_payload)}\n\n".encode()
        sequence_number += 1

        # now start the content part
        event_payload = {
            "type": "response.content_part.added",
            "sequence_number": sequence_number,
            "part": {"type": "output_text", "annotations": [], "text": ""},
        }
        yield b"event: response.content_part.added\n"
        yield f"data: {json.dumps(event_payload)}\n\n".encode()
        sequence_number += 1

        # emit content delta events for each chunk
        overall_content = ""
        for chunk in chat_completion_stream:
            payload = chunk.model_dump()

            # last chunk, include usage if requested
            if payload.get("finish_reason"):
                if include_usage and hasattr(chunk, "usage") and chunk.usage:
                    usage_info = {
                        "prompt_tokens": chunk.usage.prompt_tokens,
                        "completion_tokens": chunk.usage.completion_tokens,
                        "total_tokens": chunk.usage.total_tokens,
                    }

            # extract content from the streamed choices
            for choice in payload.get("choices", [{}]):
                delta = choice.get("delta", {})
                content = delta.get("content", "")

                # note: there is some additional tool_call info but for now,
                #       we're just grabbing the content

                # emit content delta event
                event_payload = {
                    "type": "response.output_text.delta",
                    "sequence_number": sequence_number,
                    "item_id": payload.get("id", ""),
                    "delta": content,
                }
                yield b"event: response.output_text.delta\n"
                yield f"data: {json.dumps(event_payload)}\n\n".encode()
                sequence_number += 1
                overall_content += content

        # emit done events for the item
        event_payload = {
            "id": "last_event",
            "type": "response.output_text.done",
            "sequence_number": sequence_number,
            "text": overall_content,
            "usage": usage_info,
        }
        yield b"event: response.output_text.done\n"
        yield f"data: {json.dumps(event_payload)}\n\n".encode()
        sequence_number += 1

        event_payload = {
            "type": "response.output_item.done",
            "sequence_number": sequence_number,
        }
        yield b"event: response.output_item.done\n"
        yield f"data: {json.dumps(event_payload)}\n\n".encode()
        sequence_number += 1

        yield b"data: [DONE]\n\n"

    return StreamingResponse(_generator(), media_type="text/event-stream")


def chat_completion_to_openresponses_response(
    chat_response: ChatCompletion,
    metadata: dict[str, Any] | None = None,
) -> ResponseResource:
    """
    Convert a Chat Completions response to valid Open Responses response

    Args:
        chat_response (ChatCompletion): Non-streaming chat completion response.

    Returns:
        ResponseResource: Open Responses response format
    """
    response = _openai_chat_completion_to_openresponses(chat_response, metadata=metadata)
    return response


def _openai_chat_completion_to_openresponses(
    openai_chat: ChatCompletion,
    metadata: dict[str, Any] | None = None,
) -> ResponseResource:
    """
    Convert an OpenAI Python-Client `ChatCompletion` into an Open Responses `ResponseResource`.

    Note: This is lossy and makes a lot of assumptions (see comments). OpenAI's chat completion information
          in their SDK simply doesn't have the same data as Open Responses' `ResponseResource`, so
          we're setting a lot of things to "None" or -1 or the option from OpenResponses that makes
          the "most sense" or is the "most permissive" so clients can handle that information
          as best guess.

          This is not ideal, but this is a shim anyway. Ideally we connect to inference providers
          that either expose Open Responses directly OR OpenAI's Responses, which is much more
          compatible.
    """

    return ResponseResource(
        id=openai_chat.id,
        object=Object.response,
        # openai gives float unix timestamps
        created_at=int(openai_chat.created),
        completed_at=None,
        status="completed",
        incomplete_details=None,
        model=openai_chat.model,
        previous_response_id=None,
        instructions=None,
        output=_convert_chat_choices_outputs(openai_chat.choices),
        error=None,
        # here
        tools=_convert_chat_choices_tools(openai_chat.choices),
        # for tool_choice, provide all tools used and say the model had "auto"
        tool_choice=AllowedToolChoice(
            type=Type33.allowed_tools,
            tools=_convert_chat_choices_tools(openai_chat.choices),
            mode=ToolChoiceValueEnum.auto,
        ),
        truncation=TruncationEnum.auto,
        # going to default parallel_tool_calls to true since it seems the
        # more "permissive" option and because we don't have the data in the response
        # let's tell clients it was the more "permissive" thing done
        parallel_tool_calls=True,
        # presume text response, no verbosity is provided so set to medium
        text=TextField(format=TextResponseFormat(type=Type34(value="text")), verbosity=VerbosityEnum.medium),
        # some of these are required in Open Responses but can be None in openai python SDK,
        # so use -1 as default to indicate unknown value
        top_logprobs=-1,
        top_p=-1,
        presence_penalty=-1,
        frequency_penalty=-1,
        temperature=-1,
        max_output_tokens=None,
        reasoning=None,
        usage=_convert_usage(openai_chat_usage=getattr(openai_chat, "usage", None)),
        max_tool_calls=None,
        store=False,
        background=False,
        service_tier=str(getattr(openai_chat, "service_tier", "default")),
        metadata=metadata,
        safety_identifier=None,
        prompt_cache_key=None,
    )


def _convert_usage(openai_chat_usage: OaiCompletionUsage | None) -> Usage | None:
    if not openai_chat_usage:
        return None

    # -1 to indicate unknown
    input_tokens_details = InputTokensDetails(cached_tokens=-1)
    if openai_chat_usage.prompt_tokens_details:
        input_tokens_details = InputTokensDetails(
            cached_tokens=openai_chat_usage.prompt_tokens_details.cached_tokens or 0
        )

    # -1 to indicate unknown
    output_tokens_details = OutputTokensDetails(reasoning_tokens=-1)
    if openai_chat_usage.prompt_tokens_details:
        output_tokens_details = OutputTokensDetails(
            reasoning_tokens=openai_chat_usage.completion_tokens_details.reasoning_tokens or 0
        )

    return Usage(
        input_tokens=openai_chat_usage.prompt_tokens,
        output_tokens=openai_chat_usage.completion_tokens,
        total_tokens=openai_chat_usage.total_tokens,
        input_tokens_details=input_tokens_details,
        output_tokens_details=output_tokens_details,
    )


def _convert_chat_choices_outputs(
    openai_chat_choices: list[OaiChatChoice],
) -> list[Message | FunctionCall | FunctionCallOutput | ReasoningBody]:
    response_outputs = []

    for choice in openai_chat_choices:
        message_content = choice.message.content or ""
        response_outputs.append(
            Message(
                type="message",
                # since the index of the choice is locally unique, use it for ID
                # b/c there's no other id field available for Chat Completion choices
                id=str(choice.index),
                status=MessageStatus.completed,
                role=MessageRole.assistant,
                # the spec says that messages are to OR from the model, so the
                # text type is "input_text", even though this is output
                content=[InputTextContent(type="input_text", text=message_content)],
            )
        )

        # NOTE: ignore tool_calls, since they don't appear to contain the actual tool outputs
        # it appears the output is still in the message content (which is captured above
        # as a message). It's possible we could infer that a message content was function
        # output if there is a function call in the tool_calls, but I'm going to keep this
        # simple for now and treat all output as messages.
        #
        # The function calls that could be used are still made clear to the client
        # in the `tool_choice` field.

    return response_outputs


def _convert_chat_choices_tools(openai_chat_choices: list[OaiChatChoice]):
    # so OpenAI has a broader concept of tools than Open Responses, which
    # appears to only really define function tools right now.
    # So... ignore others?
    # TODO: If openresponses supports more than FunctionTools in the future, update this
    openresponses_tools = []
    for choice in openai_chat_choices:
        tool_calls = choice.message.tool_calls or []
        for chat_tool in tool_calls:
            if not isinstance(chat_tool, ChatCompletionMessageFunctionToolCall):
                logging.debug(f"Skipping non-function tool call: {type(chat_tool)}")
                continue

            tool = FunctionTool(
                type=Type31(value="function"),
                name=chat_tool.function.name,
                description=None,
                # this expects a dict but we get a string, doing our bets
                # to make it clear this dict key is not an actual arg
                parameters={"__openai_chat_function_args__": chat_tool.function.arguments},
                strict=None,
            )
            openresponses_tools.append(tool)

    return openresponses_tools
