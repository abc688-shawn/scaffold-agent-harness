"""使用 MockModel 的 ReAct agent 循环测试。"""
from __future__ import annotations

import asyncio
import pytest

from scaffold.models.base import Message, ToolCall, ModelResponse, Usage
from scaffold.models.mock import MockModel
from scaffold.tools.registry import ToolRegistry
from scaffold.context.window import ContextWindow
from scaffold.loop.react import ReActLoop, LoopConfig


class TestReActLoop:
    def _setup(self, script: list[ModelResponse], tools: ToolRegistry | None = None):
        model = MockModel(script=script)
        reg = tools or ToolRegistry()
        ctx = ContextWindow(system_prompt="You are a test agent.")
        config = LoopConfig(max_steps=10)
        loop = ReActLoop(model=model, tools=reg, context=ctx, config=config)
        return loop

    def test_simple_text_response(self):
        loop = self._setup([
            ModelResponse(message=Message.assistant("Hello!"), usage=Usage(10, 5)),
        ])
        result = asyncio.get_event_loop().run_until_complete(loop.run("Hi"))
        assert result.final_message == "Hello!"
        assert result.steps == 1

    def test_tool_call_then_response(self):
        reg = ToolRegistry()

        @reg.tool
        def greet(name: str) -> str:
            """向某人问好。"""
            return f"Hi, {name}!"

        script = [
            ModelResponse(
                message=Message.assistant(
                    tool_calls=[ToolCall(id="1", name="greet", arguments={"name": "Alice"})]
                ),
                usage=Usage(20, 10),
            ),
            ModelResponse(
                message=Message.assistant("I greeted Alice for you!"),
                usage=Usage(30, 15),
            ),
        ]
        loop = self._setup(script, tools=reg)
        result = asyncio.get_event_loop().run_until_complete(loop.run("Greet Alice"))
        assert result.final_message == "I greeted Alice for you!"
        assert result.steps == 2

    def test_max_steps_limit(self):
        # 模型始终返回工具调用，因此应触发最大步数限制
        reg = ToolRegistry()

        @reg.tool
        def noop() -> str:
            """什么也不做。"""
            return "ok"

        infinite_calls = [
            ModelResponse(
                message=Message.assistant(
                    tool_calls=[ToolCall(id=str(i), name="noop", arguments={})]
                ),
                usage=Usage(10, 5),
            )
            for i in range(25)
        ]
        loop = self._setup(infinite_calls, tools=reg)
        loop._config.max_steps = 5
        result = asyncio.get_event_loop().run_until_complete(loop.run("Do stuff"))
        assert result.steps <= 5

    def test_loop_detection(self):
        reg = ToolRegistry()

        @reg.tool
        def repeat() -> str:
            """重复返回相同结果。"""
            return "same"

        # 同一工具调用出现 3 次以上时应触发反思
        same_call = ModelResponse(
            message=Message.assistant(
                tool_calls=[ToolCall(id="1", name="repeat", arguments={})]
            ),
            usage=Usage(10, 5),
        )
        script = [same_call] * 5 + [
            ModelResponse(message=Message.assistant("Done after reflection"), usage=Usage(10, 5)),
        ]
        loop = self._setup(script, tools=reg)
        loop._config.loop_detect_window = 3
        result = asyncio.get_event_loop().run_until_complete(loop.run("test"))
        # 在某个时刻应该注入了一条反思消息
        assert result.steps > 3
