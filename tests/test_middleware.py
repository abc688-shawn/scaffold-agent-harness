"""Middleware 管道实现的测试。"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from scaffold.context.window import ContextWindow
from scaffold.loop.middleware import StepContext
from scaffold.models.base import ModelResponse, ToolCall, ToolResult, Usage
from scaffold.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _make_ctx(step: int = 1) -> StepContext:
    return StepContext(
        step=step,
        context=MagicMock(spec=ContextWindow),
        tools=MagicMock(spec=ToolRegistry),
        total_usage=Usage(),
    )


def _tool_result(content: str = "ok", is_error: bool = False) -> ToolResult:
    return ToolResult(tool_call_id="tc1", name="list_files", content=content, is_error=is_error)


def _tool_call(name: str = "list_files", args: dict | None = None) -> ToolCall:
    return ToolCall(id="tc1", name=name, arguments=args or {})


# ---------------------------------------------------------------------------
# ToolCallLimitMiddleware
# ---------------------------------------------------------------------------

class TestToolCallLimitMiddleware:
    @pytest.mark.asyncio
    async def test_no_warning_under_limit(self):
        from scaffold.loop.middlewares import ToolCallLimitMiddleware
        mw = ToolCallLimitMiddleware(repeat_limit=3)
        ctx = _make_ctx()
        call = _tool_call()
        result = _tool_result("file list")

        out = await mw.after_tool(ctx, call, result)
        assert out.content == "file list"
        assert not out.is_error

    @pytest.mark.asyncio
    async def test_warning_on_repeat_limit(self):
        from scaffold.loop.middlewares import ToolCallLimitMiddleware
        mw = ToolCallLimitMiddleware(repeat_limit=2)
        ctx = _make_ctx()
        call = _tool_call("list_files", {"path": "."})

        # 前两次调用：无警告
        r1 = await mw.after_tool(ctx, call, _tool_result("r1"))
        r2 = await mw.after_tool(ctx, call, _tool_result("r2"))
        assert "⚠" not in (r1.content or "")
        assert "⚠" in (r2.content or "")

    @pytest.mark.asyncio
    async def test_block_behavior_returns_error(self):
        from scaffold.loop.middlewares import ToolCallLimitMiddleware
        mw = ToolCallLimitMiddleware(repeat_limit=1, exit_behavior="block")
        ctx = _make_ctx()
        call = _tool_call()

        await mw.after_tool(ctx, call, _tool_result("first"))
        out = await mw.after_tool(ctx, call, _tool_result("second"))
        assert out.is_error is True

    @pytest.mark.asyncio
    async def test_run_limit_fires_separately(self):
        from scaffold.loop.middlewares import ToolCallLimitMiddleware
        mw = ToolCallLimitMiddleware(repeat_limit=99, run_limit=2)
        ctx = _make_ctx()

        # 不同参数时 repeat_key 不同，不会触发 repeat_limit。
        # run_limit=2 在 run_count >= 2 时触发，即从第 2 次调用开始。
        for i in range(3):
            call = _tool_call("list_files", {"path": str(i)})
            out = await mw.after_tool(ctx, call, _tool_result(f"r{i}"))
            if i >= 1:
                assert "⚠" in (out.content or "")
            else:
                assert "⚠" not in (out.content or "")

    @pytest.mark.asyncio
    async def test_different_args_not_conflated(self):
        from scaffold.loop.middlewares import ToolCallLimitMiddleware
        mw = ToolCallLimitMiddleware(repeat_limit=2)
        ctx = _make_ctx()

        call_a = _tool_call("list_files", {"path": "/a"})
        call_b = _tool_call("list_files", {"path": "/b"})

        # 每个唯一的 (name, args) 对都有独立的计数器
        r1 = await mw.after_tool(ctx, call_a, _tool_result("r1"))
        r2 = await mw.after_tool(ctx, call_b, _tool_result("r2"))
        assert "⚠" not in (r1.content or "")
        assert "⚠" not in (r2.content or "")


# ---------------------------------------------------------------------------
# RedactionMiddleware
# ---------------------------------------------------------------------------

class TestRedactionMiddleware:
    @pytest.mark.asyncio
    async def test_redacts_secret_key(self):
        from scaffold.loop.middlewares import RedactionMiddleware
        mw = RedactionMiddleware()
        ctx = _make_ctx()
        call = _tool_call()
        # api_key 模式要求密钥值长度至少 20 个字符
        secret = "ABCDEFGHIJ1234567890XYZ"
        result = _tool_result(f"api_key={secret}")

        out = await mw.after_tool(ctx, call, result)
        assert secret not in (out.content or "")
        assert "REDACTED" in (out.content or "")

    @pytest.mark.asyncio
    async def test_clean_content_unchanged(self):
        from scaffold.loop.middlewares import RedactionMiddleware
        mw = RedactionMiddleware()
        ctx = _make_ctx()
        call = _tool_call()
        result = _tool_result("hello world, no secrets here")

        out = await mw.after_tool(ctx, call, result)
        assert out is result  # same object when nothing changed


# ---------------------------------------------------------------------------
# CostTrackerMiddleware
# ---------------------------------------------------------------------------

class TestCostTrackerMiddleware:
    @pytest.mark.asyncio
    async def test_no_warning_under_threshold(self, caplog):
        from scaffold.loop.middlewares import CostTrackerMiddleware
        mw = CostTrackerMiddleware(warn_fraction=0.9)
        ctx = _make_ctx()
        ctx.max_total_tokens = 1000
        ctx.total_usage = Usage(prompt_tokens=100, completion_tokens=50)

        import logging
        with caplog.at_level(logging.WARNING):
            await mw.before_step(ctx)
        assert "budget" not in caplog.text.lower()

    @pytest.mark.asyncio
    async def test_warning_over_threshold(self, caplog):
        from scaffold.loop.middlewares import CostTrackerMiddleware
        mw = CostTrackerMiddleware(warn_fraction=0.8)
        ctx = _make_ctx()
        ctx.max_total_tokens = 1000
        ctx.total_usage = Usage(prompt_tokens=850, completion_tokens=0)

        import logging
        with caplog.at_level(logging.WARNING):
            await mw.before_step(ctx)
        assert caplog.text  # some warning was logged


# ---------------------------------------------------------------------------
# Middleware pipeline integration (via ReActLoop)
# ---------------------------------------------------------------------------

class TestMiddlewarePipeline:
    """验证 Middleware 在真实循环运行中按正确顺序触发。"""

    @pytest.mark.asyncio
    async def test_after_tool_called_for_each_tool(self):
        """after_tool 每步每次工具调用都应被调用一次。"""
        from scaffold.loop.react import LoopConfig, ReActLoop
        from scaffold.context.window import ContextWindow
        from scaffold.context.budget import TokenBudget
        from scaffold.models.mock import MockModel
        from scaffold.tools.registry import ToolRegistry

        call_log: list[str] = []

        class _TrackingMiddleware:
            async def before_step(self, ctx): pass
            async def after_llm(self, ctx, response): pass
            async def after_tool(self, ctx, call, result):
                call_log.append(call.name)
                return result
            async def after_step(self, ctx): pass

        from scaffold.models.base import ModelResponse, ToolCall as TC, Message

        script = [
            ModelResponse(message=Message.assistant(tool_calls=[TC(id="t1", name="test_tool", arguments={})])),
            ModelResponse(message=Message.assistant("done")),
        ]
        model = MockModel(script=script)

        tool_registry = ToolRegistry()

        @tool_registry.tool(name="test_tool", description="test")
        async def test_tool() -> str:
            return "result"

        context = ContextWindow(system_prompt="system", budget=TokenBudget())
        loop = ReActLoop(
            model=model,
            tools=tool_registry,
            context=context,
            config=LoopConfig(max_steps=5),
            middlewares=[_TrackingMiddleware()],
        )

        result = await loop.run("run the tool")
        assert result.final_message == "done"
        assert call_log == ["test_tool"]
