"""Tests for the MockModel and base message types."""
import asyncio
import pytest

from scaffold.models.base import Message, ToolCall, ToolResult, ModelResponse, Usage, Role
from scaffold.models.mock import MockModel


class TestMessageTypes:
    def test_system_message(self):
        m = Message.system("You are helpful")
        assert m.role == Role.SYSTEM
        assert m.content == "You are helpful"

    def test_user_message(self):
        m = Message.user("Hello")
        assert m.role == Role.USER

    def test_assistant_with_tool_calls(self):
        tc = ToolCall(id="1", name="read_file", arguments={"path": "test.txt"})
        m = Message.assistant(tool_calls=[tc])
        assert m.tool_calls is not None
        assert len(m.tool_calls) == 1
        assert m.tool_calls[0].name == "read_file"

    def test_tool_result_message(self):
        tr = ToolResult(tool_call_id="1", name="read_file", content="file content")
        m = Message.tool_result(tr)
        assert m.role == Role.TOOL
        assert m.content == "file content"
        assert m.tool_call_id == "1"

    def test_tool_call_arguments_json(self):
        tc = ToolCall(id="1", name="test", arguments={"key": "value", "num": 42})
        j = tc.arguments_json()
        assert '"key"' in j
        assert '"value"' in j


class TestUsage:
    def test_total_tokens(self):
        u = Usage(prompt_tokens=100, completion_tokens=50)
        assert u.total_tokens == 150


class TestMockModel:
    def test_scripted_responses(self):
        script = [
            ModelResponse(message=Message.assistant("First"), usage=Usage(10, 5)),
            ModelResponse(message=Message.assistant("Second"), usage=Usage(10, 5)),
        ]
        model = MockModel(script=script)

        loop = asyncio.get_event_loop()
        r1 = loop.run_until_complete(model.chat([Message.user("hi")]))
        assert r1.message.content == "First"

        r2 = loop.run_until_complete(model.chat([Message.user("hi")]))
        assert r2.message.content == "Second"

    def test_script_exhausted(self):
        model = MockModel(script=[])
        loop = asyncio.get_event_loop()
        r = loop.run_until_complete(model.chat([Message.user("hi")]))
        assert "exhausted" in r.message.content.lower()

    def test_call_log(self):
        model = MockModel(script=[
            ModelResponse(message=Message.assistant("ok")),
        ])
        loop = asyncio.get_event_loop()
        loop.run_until_complete(model.chat([Message.user("test")]))
        assert len(model.call_log) == 1
        assert model.call_log[0][0].content == "test"
