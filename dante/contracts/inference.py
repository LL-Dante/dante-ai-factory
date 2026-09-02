"""Provider-independent inference messages and envelopes (P1)."""
from __future__ import annotations

import json
from typing import Annotated, Literal

from pydantic import Field, JsonValue, StrictInt, model_validator

from dante.contracts import ModelRef, StrictModel


class ToolCall(StrictModel):
    call_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: dict[str, JsonValue]

    @model_validator(mode='after')
    def finite_arguments(self):
        json.dumps(self.arguments, allow_nan=False)
        return self


class TextMessage(StrictModel):
    role: Literal['system', 'user']
    content: str


class AssistantMessage(StrictModel):
    role: Literal['assistant'] = 'assistant'
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()

    @model_validator(mode='after')
    def unique_calls(self):
        ids = [call.call_id for call in self.tool_calls]
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate tool call IDs')
        return self


class ToolResult(StrictModel):
    role: Literal['tool'] = 'tool'
    call_id: str = Field(min_length=1)
    content: str


Message = Annotated[TextMessage | AssistantMessage | ToolResult, Field(discriminator='role')]


class ToolDefinition(StrictModel):
    name: str = Field(min_length=1)
    description: str = ''
    parameters: dict[str, JsonValue]

    @model_validator(mode='after')
    def object_schema(self):
        json.dumps(self.parameters, allow_nan=False)
        if self.parameters.get('type') != 'object':
            raise ValueError('Tool parameters must describe an object')
        return self


class InferenceRequest(StrictModel):
    model: ModelRef
    messages: tuple[Message, ...] = Field(min_length=1)
    tools: tuple[ToolDefinition, ...] = ()
    temperature: float = Field(default=0, ge=0, le=2, allow_inf_nan=False)
    max_output_tokens: int | None = Field(default=None, gt=0)
    task_id: str | None = None
    trace_id: str | None = None

    @model_validator(mode='after')
    def continuation_ids(self):
        seen, pending = set(), set()
        for message in self.messages:
            if isinstance(message, ToolResult):
                if message.call_id not in pending:
                    raise ValueError('Unmatched or repeated tool result')
                pending.remove(message.call_id)
            else:
                if pending:
                    raise ValueError('Missing tool results before next message')
                if isinstance(message, AssistantMessage):
                    for call in message.tool_calls:
                        if call.call_id in seen:
                            raise ValueError('Reused tool call ID')
                        seen.add(call.call_id)
                        pending.add(call.call_id)
        if pending:
            raise ValueError('Missing tool results')
        names = [tool.name for tool in self.tools]
        if len(names) != len(set(names)):
            raise ValueError('Duplicate tool definitions')
        return self


class Usage(StrictModel):
    input_tokens: StrictInt | None = Field(default=None, ge=0)
    output_tokens: StrictInt | None = Field(default=None, ge=0)
    total_tokens: StrictInt | None = Field(default=None, ge=0)


class InferenceResponse(StrictModel):
    model: ModelRef
    assistant_message: AssistantMessage
    finish_reason: Literal['stop', 'tool_calls', 'length', 'content_filter', 'unknown'] = 'unknown'
    usage: Usage = Field(default_factory=Usage)
    cost: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    fallback: bool = False

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        return self.assistant_message.tool_calls

    @property
    def content(self) -> str:
        """Compatibility accessor for existing foundation consumers."""
        return self.assistant_message.content or ''

    @property
    def model_id(self) -> str:
        return self.model.model_id

    @property
    def provider_id(self) -> str:
        return self.model.provider_id
