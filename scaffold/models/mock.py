"""Mock model for testing and eval replay.

Usage:
    model = MockModel(script=[
        ModelResponse(message=Message.assistant("Hello!")),
        ModelResponse(message=Message.assistant(
            tool_calls=[ToolCall(id="1", name="list_files", arguments={"path": "."})]
        )),
    ])
"""
from __future__ import annotations

from typing import Any, AsyncIterator, Sequence

from scaffold.models.base import (
    ChatModel,
    Message,
    ModelResponse,
    Usage,
)


class MockModel(ChatModel):
    """Returns pre-scripted responses in order.  Useful for deterministic tests."""

    def __init__(self, script: list[ModelResponse] | None = None) -> None:
        self._script = list(script or [])
        self._cursor = 0
        self.call_log: list[list[Message]] = []

    def enqueue(self, *responses: ModelResponse) -> None:
        self._script.extend(responses)

    async def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        self.call_log.append(list(messages))
        if self._cursor >= len(self._script):
            return ModelResponse(
                message=Message.assistant("(MockModel: script exhausted)"),
                usage=Usage(),
                finish_reason="stop",
            )
        resp = self._script[self._cursor]
        self._cursor += 1
        return resp

    async def chat_stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ModelResponse]:
        resp = await self.chat(messages, tools, temperature, max_tokens, **kwargs)
        yield resp
