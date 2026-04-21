"""OpenAI 兼容适配器。

适用于任何暴露 OpenAI 风格 `/chat/completions` 端点的提供商，
例如 DeepSeek、GLM（智谱）、MiniMax M2、Moonshot 等。

用法：
    model = OpenAICompatModel(
        api_key="sk-xxx",
        base_url="https://api.deepseek.com/v1",
        model="deepseek-reasoner",
    )
"""
from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Sequence

from openai import AsyncOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
)

from scaffold.models.base import (
    ChatModel,
    Message,
    ModelResponse,
    ToolCall,
    Usage,
)

logger = logging.getLogger(__name__)

# 来自 openai SDK 的可重试 HTTP 错误类型
_RETRIABLE = (
    Exception,  # 下面会进一步收窄
)

def _is_retriable(exc: BaseException) -> bool:
    """如果是值得重试的瞬时错误，则返回 True。"""
    from openai import (
        APITimeoutError,
        APIConnectionError,
        RateLimitError,
        InternalServerError,
    )
    return isinstance(exc, (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError))


class OpenAICompatModel(ChatModel):
    """适配 OpenAI 兼容 chat/completions API 的模型适配器。"""

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model: str = "deepseek-reasoner",
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        self._model = model
        self._max_retries = max_retries
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,  # 重试由 tenacity 统一处理
        )

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def _to_openai_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
        """将中立 Message 转换为 OpenAI 线协议格式。"""
        out: list[dict[str, Any]] = []
        for m in messages:
            d: dict[str, Any] = {"role": m.role.value}
            if m.content is not None:
                d["content"] = m.content
            if m.tool_calls:
                d["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments_json(),
                        },
                    }
                    for tc in m.tool_calls
                ]
            if m.tool_call_id:
                d["tool_call_id"] = m.tool_call_id
            if m.name:
                d["name"] = m.name
            out.append(d)
        return out

    @staticmethod
    def _parse_tool_calls(raw_tool_calls: list[Any] | None) -> list[ToolCall] | None:
        if not raw_tool_calls:
            return None
        result = []
        for tc in raw_tool_calls:
            func = tc.function
            try:
                args = json.loads(func.arguments) if func.arguments else {}
            except json.JSONDecodeError:
                args = {"_raw": func.arguments}
            result.append(ToolCall(id=tc.id, name=func.name, arguments=args))
        return result

    # ------------------------------------------------------------------
    # 对外公开接口
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception(_is_retriable),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    async def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        api_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": self._to_openai_messages(messages),
            "temperature": temperature,
        }
        if max_tokens is not None:
            api_kwargs["max_tokens"] = max_tokens
        if tools:
            api_kwargs["tools"] = tools
        api_kwargs.update(kwargs)

        logger.debug("LLM request: model=%s messages=%d tools=%s",
                      self._model, len(messages), len(tools) if tools else 0)

        raw = await self._client.chat.completions.create(**api_kwargs)
        choice = raw.choices[0]

        tool_calls = self._parse_tool_calls(
            getattr(choice.message, "tool_calls", None)
        )

        msg = Message.assistant(
            content=choice.message.content,
            tool_calls=tool_calls,
        )

        usage = Usage(
            prompt_tokens=raw.usage.prompt_tokens if raw.usage else 0,
            completion_tokens=raw.usage.completion_tokens if raw.usage else 0,
        )

        return ModelResponse(
            message=msg,
            usage=usage,
            finish_reason=choice.finish_reason,
            raw=raw,
        )

    async def chat_stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ModelResponse]:
        api_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": self._to_openai_messages(messages),
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens is not None:
            api_kwargs["max_tokens"] = max_tokens
        if tools:
            api_kwargs["tools"] = tools
        api_kwargs.update(kwargs)

        stream = await self._client.chat.completions.create(**api_kwargs)

        # 跨多个 chunk 累积工具调用
        accumulated_tool_calls: dict[int, dict[str, Any]] = {}
        accumulated_content = ""
        finish_reason = None
        prompt_tokens = 0
        completion_tokens = 0

        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            finish_reason = chunk.choices[0].finish_reason or finish_reason

            if delta.content:
                accumulated_content += delta.content

            # 累积流式返回的工具调用参数片段
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in accumulated_tool_calls:
                        accumulated_tool_calls[idx] = {
                            "id": tc_delta.id or "",
                            "name": tc_delta.function.name if tc_delta.function and tc_delta.function.name else "",
                            "arguments": "",
                        }
                    acc = accumulated_tool_calls[idx]
                    if tc_delta.id:
                        acc["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            acc["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            acc["arguments"] += tc_delta.function.arguments

            if chunk.usage:
                prompt_tokens = chunk.usage.prompt_tokens
                completion_tokens = chunk.usage.completion_tokens

        # 构造最终响应
        tool_calls = None
        if accumulated_tool_calls:
            tool_calls = []
            for _idx in sorted(accumulated_tool_calls):
                acc = accumulated_tool_calls[_idx]
                try:
                    args = json.loads(acc["arguments"]) if acc["arguments"] else {}
                except json.JSONDecodeError:
                    args = {"_raw": acc["arguments"]}
                tool_calls.append(ToolCall(id=acc["id"], name=acc["name"], arguments=args))

        msg = Message.assistant(
            content=accumulated_content or None,
            tool_calls=tool_calls,
        )
        yield ModelResponse(
            message=msg,
            usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
            finish_reason=finish_reason,
        )
