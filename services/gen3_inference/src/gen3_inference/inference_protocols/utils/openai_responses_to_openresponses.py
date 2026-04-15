"""
TODO: FIXME: Clean this up, break out some more helpers
"""

import json
from collections.abc import AsyncGenerator

from fastapi.responses import StreamingResponse
from openai import Stream
from openai.types.responses import (
    FunctionTool as OaiFunctionTool,
)
from openai.types.responses import Response as OpenAIResponse
from openai.types.responses import (
    ResponseFormatTextJSONSchemaConfig,
    ResponseStreamEvent,
    ResponseTextConfig,
)
from openai.types.responses import (
    ResponseOutputItem as OaiResponseOutputItem,
)
from openai.types.responses import (
    Tool as OaiTool,
)
from openai.types.responses import (
    ToolChoiceAllowed as OaiToolChoiceAllowed,
)
from openai.types.responses import (
    ToolChoiceFunction as OaiToolChoiceFunction,
)
from openai.types.responses.response import ToolChoice as OaiToolChoice
from openai.types.shared import Reasoning as OaiReasoning
from openresponses_types import (
    AllowedToolChoice,
    Any,
    Error,
    FunctionCall,
    FunctionCallOutput,
    FunctionCallOutputStatusEnum,
    FunctionCallStatus,
    FunctionTool,
    FunctionToolChoice,
    ImageDetail,
    IncompleteDetails,
    InputFileContent,
    InputImageContent,
    InputTextContent,
    JsonObjectResponseFormat,
    JsonSchemaResponseFormat,
    Message,
    MessageRole,
    MessageStatus,
    Object,
    Reasoning,
    ReasoningBody,
    ReasoningEffortEnum,
    ReasoningSummaryContentParam,
    ReasoningSummaryEnum,
    ResponseResource,
    TextField,
    TextResponseFormat,
    ToolChoiceValueEnum,
    TruncationEnum,
    Type1,
    Type31,
    Type33,
    Type34,
    Type35,
    Type36,
    Usage,
    VerbosityEnum,
)

from gen3_inference.config import logging


def openai_streaming_response_to_openresponses(
    openai_response_stream: Stream[ResponseStreamEvent],
    metadata: dict[str, Any] | None = None,
):
    # TODO: add metadata to some event?
    async def _generator() -> AsyncGenerator[bytes]:
        """
        Yield each event as an Open Responses Event
        """
        for event in openai_response_stream:
            payload = event.model_dump()

            # emit the event header and data
            event_type = payload["type"]
            yield f"event: {event_type}\n".encode()
            json_text = json.dumps(payload)

            # For now, just use whatever is provided.
            # we may need more explicit conversion to
            # only Open Responses types... TODO
            yield f"data: {json_text}\n\n".encode()

        yield b"data: [DONE]\n\n"

    # return the generator wrapped in a StreamingResponse
    return StreamingResponse(_generator(), media_type="text/event-stream")


def openai_response_to_openresponses(
    openai_response: OpenAIResponse,
    metadata: dict[str, Any] | None = None,
) -> ResponseResource:
    """
    Convert an OpenAI Python-Client `Response` into an Open Responses `ResponseResource`.
    """
    metadata = metadata or {}
    return ResponseResource(
        id=openai_response.id,
        object=Object.response,
        # openai gives float unix timestamps
        created_at=int(openai_response.created_at),
        completed_at=int(openai_response.completed_at) if openai_response.completed_at else None,
        status=openai_response.status if openai_response.status else "unknown",
        incomplete_details=IncompleteDetails(**openai_response.incomplete_details.model_dump())
        if openai_response.incomplete_details
        else None,
        model=openai_response.model,
        previous_response_id=openai_response.previous_response_id,
        # TODO: do we need to handle non-strings?
        instructions=openai_response.instructions if isinstance(openai_response.instructions, str) else None,
        output=_convert_response_outputs(openai_response.output),
        error=Error(**openai_response.error.model_dump()) if openai_response.error else None,
        # ignore type issue b/c openresponses_types expects a list of Tools
        # but FunctionTool is the only tool
        # FIXME: This may break... we may need to rewrite openresponses_types
        tools=_convert_tools(openai_response.tools),  # type: ignore
        tool_choice=_convert_tool_choice(openai_response.tool_choice),
        truncation=TruncationEnum.auto if openai_response.truncation == "auto" else TruncationEnum.disabled,
        parallel_tool_calls=openai_response.parallel_tool_calls,
        text=_convert_text(openai_response.text),
        # some of these are required in Open Responses but can be None in openai python SDK,
        # so use -1 as default to indicate unknown value
        top_p=openai_response.top_p or -1,
        presence_penalty=-1,
        frequency_penalty=-1,
        top_logprobs=openai_response.top_logprobs or -1,
        temperature=openai_response.temperature or -1,
        reasoning=_convert_reasoning(openai_response.reasoning),
        usage=Usage(**openai_response.usage.model_dump()) if openai_response.usage else None,
        max_output_tokens=openai_response.max_output_tokens,
        max_tool_calls=openai_response.max_tool_calls,
        store=False,
        background=bool(openai_response.background),
        service_tier=openai_response.service_tier if openai_response.service_tier else "default",
        metadata=(openai_response.metadata or {}).update(metadata),
        safety_identifier=openai_response.safety_identifier,
        prompt_cache_key=openai_response.prompt_cache_key,
    )


def _convert_response_outputs(
    openai_outputs: list[OaiResponseOutputItem],
) -> list[Message | FunctionCall | FunctionCallOutput | ReasoningBody]:
    response_outputs = []

    for item in openai_outputs:
        if item.type == "message":
            # the spec says that messages are to OR from the model, so the
            # text type is "input_text", even though this is output
            content_list = [
                InputTextContent(type="input_text", text=content.text)
                for content in item.content
                if content.type == "output_text"
            ]

            status = MessageStatus.incomplete
            if item.status == "in_progress":
                status = MessageStatus.in_progress
            elif item.status == "completed":
                status = MessageStatus.completed
            elif item.status == "incomplete":
                status = MessageStatus.incomplete
            else:
                logging.debug(f"unknown message status: {item.status}, setting `incomplete`")

            response_outputs.append(
                Message(
                    type="message",
                    id=item.id,
                    status=status,
                    role=MessageRole.assistant,
                    content=list(content_list),
                )
            )
        elif item.type == "function_call":
            status = FunctionCallStatus.incomplete
            if item.status == "in_progress":
                status = FunctionCallStatus.in_progress
            elif item.status == "completed":
                status = FunctionCallStatus.completed
            elif item.status == "incomplete":
                status = FunctionCallStatus.incomplete
            else:
                logging.debug(f"unknown function call status: {item.status}, setting `incomplete`")

            response_outputs.append(
                FunctionCall(
                    type="function_call",
                    id=item.call_id,
                    call_id=",",
                    name=item.name,
                    arguments=item.arguments,
                    status=status,
                )
            )

        elif item.type == "function_call_output":
            output_list = []

            for output_item in item.output:
                if isinstance(output_item, str):
                    output_list.append(output_item)
                elif output_item.type == "input_file":
                    output_list.append(
                        InputFileContent(
                            type="input_file",
                            filename=output_item.filename,
                            file_url=output_item.file_url,
                        )
                    )
                elif output_item.type == "input_text":
                    output_list.append(
                        InputTextContent(
                            type="input_text",
                            text=output_item.text,
                        )
                    )
                elif output_item.type == "input_image":
                    output_list.append(
                        InputImageContent(
                            type="input_image",
                            image_url=output_item.image_url,
                            # open responses has no enum for original, so set to auto in that case
                            detail=ImageDetail(output_item.detail if output_item.detail != "original" else "auto"),
                        )
                    )

            status = FunctionCallOutputStatusEnum.incomplete
            if item.status == "in_progress":
                status = FunctionCallOutputStatusEnum.in_progress
            elif item.status == "completed":
                status = FunctionCallOutputStatusEnum.completed
            elif item.status == "incomplete":
                status = FunctionCallOutputStatusEnum.incomplete
            else:
                logging.debug(f"unknown function call status: {item.status}, setting `incomplete`")

            response_outputs.append(
                FunctionCallOutput(
                    type="function_call_output",
                    id=item.call_id,
                    call_id=",",
                    output=output_list,
                    status=status,
                )
            )
        elif item.type == "reasoning":
            # weirdly, the spec says that messages are to OR from the model, so the
            # text type is "input_text", even though this is output
            content_list = None
            if item.content:
                content_list = [
                    InputTextContent(type="input_text", text=content.text) for content in item.content if item.content
                ]

            summary_list = [
                ReasoningSummaryContentParam(type=Type1(value="summary_text"), text=content.text)
                for content in item.summary
            ]

            response_outputs.append(
                ReasoningBody(
                    type="reasoning",
                    id=item.id,
                    content=content_list,
                    summary=summary_list,
                    encrypted_content=item.encrypted_content,
                )
            )
        else:
            logging.debug(f"Unknown response item type: {item.type}. Ignoring.")

    return response_outputs


def _convert_text(openai_text: ResponseTextConfig | None) -> TextField:

    # default assuming text and medium verbosity
    format = TextResponseFormat(type=Type34(value="text"))
    verbosity = VerbosityEnum.medium

    if openai_text:
        if openai_text.format:
            openai_format_type = openai_text.format.type
            if openai_format_type == "text":
                format = TextResponseFormat(type=Type34(value="text"))
            elif openai_format_type == "json_schema":
                # type ignored / suppressed b/c pyright can't figure it out but this
                # is the right type
                openai_format: ResponseFormatTextJSONSchemaConfig = openai_text.format  # type: ignore
                format = JsonSchemaResponseFormat(
                    type=Type36(value="json_schema"),
                    name=openai_format.name,
                    description=openai_format.description,
                    # ignore type below b/c schema_ is incorrectly listed as None
                    # and spec requires an object for the schema
                    schema_=openai_format.model_json_schema(),  # type: ignore
                    strict=bool(openai_format.strict),
                )
            elif openai_format_type == "json":
                format = JsonObjectResponseFormat(type=Type35(value="json_object"))

        if openai_text.verbosity == "low":
            verbosity = VerbosityEnum.low
        elif openai_text.verbosity == "medium":
            verbosity = VerbosityEnum.medium
        elif openai_text.verbosity == "high":
            verbosity = VerbosityEnum.high
        else:
            # default to medium
            verbosity = VerbosityEnum.medium

    text = TextField(format=format, verbosity=verbosity)

    return text


def _convert_reasoning(openai_response_reasoning: OaiReasoning | None) -> Reasoning | None:
    if not openai_response_reasoning:
        return None

    effort = None

    if openai_response_reasoning.effort == "none":
        effort = ReasoningEffortEnum.none
    if openai_response_reasoning.effort == "minimal":
        # NOTE: Open Responses has no concept of "minimal", so convert to "low"
        effort = ReasoningEffortEnum.low
    if openai_response_reasoning.effort == "low":
        effort = ReasoningEffortEnum.low
    elif openai_response_reasoning.effort == "medium":
        effort = ReasoningEffortEnum.medium
    elif openai_response_reasoning.effort == "high":
        effort = ReasoningEffortEnum.high
    elif openai_response_reasoning.effort == "xhigh":
        effort = ReasoningEffortEnum.xhigh

    summary = None
    if openai_response_reasoning.summary == "concise":
        summary = ReasoningSummaryEnum.concise
    elif openai_response_reasoning.summary == "detailed":
        summary = ReasoningSummaryEnum.detailed
    elif openai_response_reasoning.summary == "auto":
        summary = ReasoningSummaryEnum.auto

    return Reasoning(effort=effort, summary=summary)


def _convert_tools(tools: list[OaiTool]) -> list[FunctionTool]:
    # so OpenAI has a broader concept of tools than Open Responses, which
    # appears to only really define function tools right now.
    # So... ignore others?
    output_tools = []
    for tool in tools:
        if isinstance(tool, OaiFunctionTool):
            output_tools.append(
                FunctionTool(
                    type=Type31(value="function"),
                    name=getattr(tool, "name", "unknown"),
                    description=getattr(tool, "description", ""),
                    parameters=getattr(tool, "parameters", {}),
                    strict=getattr(tool, "strict", True),
                )
            )
        else:
            logging.debug(f"Skipping non-function tool: {tool}")

    return output_tools


def _convert_tool_choice(
    oai_tool_choice: OaiToolChoice,
) -> AllowedToolChoice | FunctionToolChoice | ToolChoiceValueEnum:

    if isinstance(oai_tool_choice, str):
        return ToolChoiceValueEnum(value=oai_tool_choice)

    elif isinstance(oai_tool_choice, OaiToolChoiceFunction):
        return FunctionToolChoice(type=Type31(value="function"), name=oai_tool_choice.name)

    elif isinstance(oai_tool_choice, OaiToolChoiceAllowed):
        # so OpenAI has a broader concept of tools than Open Responses, which
        # appears to only really define function tools right now.
        # So... ignore others?
        # TODO: If openresponses supports more than FunctionTools in the future, update this
        output_tools = []
        for tool in oai_tool_choice.tools:
            if tool.get("type") == "function":
                output_tools.append(
                    FunctionToolChoice(type=Type31(value="function"), name=str(tool.get("name", "unknown")))
                )
            else:
                logging.debug(f"Skipping non-function tool: {tool}")

        tool_choice = AllowedToolChoice(
            type=Type33(value="allowed_tools"), tools=output_tools, mode=ToolChoiceValueEnum(oai_tool_choice.mode)
        )
        return tool_choice

    raise (Exception(f"Unknown Tool Choice Type: {oai_tool_choice}"))
