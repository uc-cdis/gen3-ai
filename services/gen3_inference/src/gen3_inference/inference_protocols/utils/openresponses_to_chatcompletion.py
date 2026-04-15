from typing import Any

from openresponses_types import (
    FunctionCall,
    FunctionCallOutput,
    Message,
    ReasoningBody,
    ResponseResource,
    Usage,
)


def _usage_to_dict(usage: Usage) -> dict[str, int]:
    """Convert the OpenResponses Usage object to the Chat-Completions usage dict."""
    return {
        "prompt_tokens": usage.input_tokens,
        "completion_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
    }


def _convert_output_items(
    items: list[Message | FunctionCall | FunctionCallOutput | ReasoningBody],
) -> dict[str, Any]:
    """
    Convert the list of OpenResponses output items into a single Chat-Completions
    message structure.  The function supports one function call per message
    (Chat-Completions only allows a single function call).  If multiple function
    calls are present they are collapsed into a single tool_calls array.
    """
    assistant_content_parts: list[str] = []
    function_call: dict[str, Any] | None = None
    tool_calls: list[dict[str, Any]] = []

    for item in items:
        if isinstance(item, Message):
            # Concatenate all text parts into a single string
            content = "".join(part.text for part in item.content if hasattr(part, "text"))
            assistant_content_parts.append(content)

        elif isinstance(item, FunctionCall):
            # Chat-Completions supports only one function_call per message
            function_call = {
                "name": item.name,
                "arguments": item.arguments,
            }

        elif isinstance(item, FunctionCallOutput):
            # Convert to a tool call – Chat-Completions uses tool_calls
            tool_calls.append(
                {
                    "id": item.call_id,
                    "type": "function",
                    "function": {
                        "name": item.name,
                        "arguments": item.arguments,
                        "output": item.output,
                    },
                }
            )

        elif isinstance(item, ReasoningBody):
            # Append reasoning text to the assistant content
            reasoning_text = "".join(part.text for part in item.content if hasattr(part, "text"))
            assistant_content_parts.append(reasoning_text)

    message: dict[str, Any] = {"role": "assistant"}
    if assistant_content_parts:
        message["content"] = "".join(assistant_content_parts)

    if function_call:
        message["function_call"] = function_call

    if tool_calls:
        message["tool_calls"] = tool_calls

    return message


def openresponses_to_chatcompletion(
    resp: ResponseResource,
) -> dict[str, Any]:
    """
    Convert an OpenResponses ResponseResource into a dict that matches the
    OpenAI Chat Completions API response format

    Args:
        resp (ResponseResource): The OpenResponses response to convert

    Returns:
        Dict[str, Any]: A dictionary that can be returned from a FastAPI endpoint or
            sent directly to the Chat Completions endpoint
    """
    chat_resp: dict[str, Any] = {
        "id": resp.id,
        "object": "chat.completion",
        "created": resp.created_at,
        "model": resp.model,
    }

    if resp.error:
        chat_resp["choices"] = [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"Error: {resp.error.message}",
                },
                "finish_reason": "error",
            }
        ]
        chat_resp["usage"] = _usage_to_dict(resp.usage) if resp.usage else {}
        return chat_resp

    message = _convert_output_items(resp.output)

    chat_resp["choices"] = [
        {
            "index": 0,
            "message": message,
            "finish_reason": "stop",
        }
    ]

    chat_resp["usage"] = _usage_to_dict(resp.usage) if resp.usage else {}

    return chat_resp
