"""Base abstractions for LLM adapters.

Design decision: we define our own Message / ToolCall dataclasses so the rest of
Scaffold never touches provider-specific formats.  Each concrete adapter is
responsible for converting *to* the provider wire format and *from* it back into
these neutral types.
"""
from __future__ import annotations

import abc
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Sequence


# ---------------------------------------------------------------------------
# Neutral message types
# ---------------------------------------------------------------------------

class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ToolCall:
    """A single tool invocation requested by the model."""
    id: str
    name: str
    arguments: dict[str, Any]

    def arguments_json(self) -> str:
        return json.dumps(self.arguments, ensure_ascii=False)


@dataclass
class ToolResult:
    """The result we send back after executing a tool."""
    tool_call_id: str
    name: str
    content: str
    is_error: bool = False


@dataclass
class Message:
    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    # ---- convenience constructors ----
    @classmethod
    def system(cls, content: str) -> "Message":
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(cls, content: str | None = None,
                  tool_calls: list[ToolCall] | None = None) -> "Message":
        return cls(role=Role.ASSISTANT, content=content, tool_calls=tool_calls)

    @classmethod
    def tool_result(cls, result: ToolResult) -> "Message":
        return cls(
            role=Role.TOOL,
            content=result.content,
            tool_call_id=result.tool_call_id,
            name=result.name,
        )


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class ModelResponse:
    """Normalized response from any LLM provider."""
    message: Message
    usage: Usage = field(default_factory=Usage)
    finish_reason: str | None = None
    raw: Any = None  # original provider response, for debugging


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class ChatModel(abc.ABC):
    """Unified interface every LLM adapter must implement."""

    @abc.abstractmethod
    async def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Single-turn (non-streaming) chat completion."""

    async def chat_stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ModelResponse]:
        """Streaming variant — yields partial deltas.

        Default implementation falls back to non-streaming.
        """
        resp = await self.chat(messages, tools, temperature, max_tokens, **kwargs)
        yield resp
