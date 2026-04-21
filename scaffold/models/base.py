"""LLM 适配器的基础抽象。

设计决策：我们定义了自己的 Message / ToolCall 数据类，这样 Scaffold
其余部分就不需要接触特定于提供商的格式。每个具体适配器都负责在提供商的
线协议格式与这些中立类型之间进行双向转换。
"""
from __future__ import annotations

import abc
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Sequence


# ---------------------------------------------------------------------------
# 中立消息类型
# ---------------------------------------------------------------------------

class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ToolCall:
    """模型请求的一次工具调用。"""
    id: str
    name: str
    arguments: dict[str, Any]

    def arguments_json(self) -> str:
        return json.dumps(self.arguments, ensure_ascii=False)


@dataclass
class ToolResult:
    """工具执行后回传给模型的结果。"""
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

    # ---- 便捷构造方法 ----
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
    """来自任意 LLM 提供商的标准化响应。"""
    message: Message
    usage: Usage = field(default_factory=Usage)
    finish_reason: str | None = None
    raw: Any = None  # 提供商返回的原始响应，便于调试


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------

class ChatModel(abc.ABC):
    """每个 LLM 适配器都必须实现的统一接口。"""

    @abc.abstractmethod
    async def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """单轮对话的非流式补全。"""

    async def chat_stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ModelResponse]:
        """流式变体，会逐步产出增量结果。

        默认实现会回退到非流式调用。
        """
        resp = await self.chat(messages, tools, temperature, max_tokens, **kwargs)
        yield resp
