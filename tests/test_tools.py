"""Tests for Tool Runtime: registry, schema generation, error handling."""
import asyncio
import pytest

from scaffold.tools.registry import ToolRegistry
from scaffold.tools.errors import ToolError, ToolErrorCode
from scaffold.tools.schema import schema_from_function
from scaffold.models.base import ToolCall


class TestSchemaGeneration:
    def test_basic_function(self):
        def greet(name: str, loud: bool = False) -> str:
            """Say hello.

            name: Person's name.
            loud: Whether to shout.
            """
            return f"Hello, {name}!"

        schema = schema_from_function(greet)
        assert schema.name == "greet"
        assert "hello" in schema.description.lower()
        assert "name" in schema.parameters["properties"]
        assert "name" in schema.parameters["required"]
        assert "loud" not in schema.parameters["required"]

    def test_openai_format(self):
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        schema = schema_from_function(add)
        oai = schema.to_openai()
        assert oai["type"] == "function"
        assert oai["function"]["name"] == "add"
        assert oai["function"]["parameters"]["properties"]["a"]["type"] == "integer"


class TestToolRegistry:
    def test_register_and_execute(self):
        reg = ToolRegistry()

        @reg.tool
        def echo(text: str) -> str:
            """Echo the input."""
            return text

        assert "echo" in reg.list_names()

        call = ToolCall(id="1", name="echo", arguments={"text": "hello"})
        result = asyncio.get_event_loop().run_until_complete(reg.execute(call))
        assert result.content == "hello"
        assert not result.is_error

    def test_unknown_tool(self):
        reg = ToolRegistry()
        call = ToolCall(id="1", name="nonexistent", arguments={})
        result = asyncio.get_event_loop().run_until_complete(reg.execute(call))
        assert result.is_error
        assert "not found" in result.content.lower()

    def test_tool_error_propagation(self):
        reg = ToolRegistry()

        @reg.tool
        def fail_tool(x: str) -> str:
            """Always fails."""
            raise ToolError(ToolErrorCode.NOT_FOUND, "File not found", hint="Check the path.")

        call = ToolCall(id="1", name="fail_tool", arguments={"x": "test"})
        result = asyncio.get_event_loop().run_until_complete(reg.execute(call))
        assert result.is_error
        assert "not_found" in result.content
        assert "Check the path" in result.content

    def test_openai_tools_output(self):
        reg = ToolRegistry()

        @reg.tool
        def my_tool(a: str, b: int = 0) -> str:
            """Do something."""
            return "ok"

        tools = reg.to_openai_tools()
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "my_tool"


class TestToolErrors:
    def test_error_formatting(self):
        err = ToolError(ToolErrorCode.NOT_FOUND, "File X not found")
        msg = err.for_model()
        assert "[not_found]" in msg
        assert "File X not found" in msg
        assert "Hint:" in msg  # default hint

    def test_custom_hint(self):
        err = ToolError(ToolErrorCode.INTERNAL_ERROR, "Oops", hint="Try again later")
        msg = err.for_model()
        assert "Try again later" in msg
