"""冒烟测试 —— 验证 DeepSeek API 连通性与工具调用能力。

用法：
    python scripts/smoke_test.py
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scaffold.models.openai_compat import OpenAICompatModel
from scaffold.models.base import Message
from scaffold.tools.registry import ToolRegistry

registry = ToolRegistry()


@registry.tool
def get_weather(city: str) -> str:
    """获取某个城市的当前天气。

    city: 需要查询天气的城市名。
    """
    return f"The weather in {city} is sunny, 22°C."


async def test_basic_chat(model: OpenAICompatModel) -> bool:
    """测试基础对话补全。"""
    print("  [1] Basic chat...", end=" ", flush=True)
    try:
        response = await model.chat(
            messages=[Message.user("Say hello in exactly 3 words.")],
        )
        print(f"OK — '{response.message.content[:80]}'")
        print(f"      tokens: {response.usage.prompt_tokens}+{response.usage.completion_tokens}")
        return True
    except Exception as e:
        print(f"FAIL — {e}")
        return False


async def test_tool_call(model: OpenAICompatModel) -> bool:
    """测试工具调用。"""
    print("  [2] Tool call...", end=" ", flush=True)
    try:
        response = await model.chat(
            messages=[Message.user("What's the weather in Beijing?")],
            tools=registry.to_openai_tools(),
        )
        if response.message.tool_calls:
            tc = response.message.tool_calls[0]
            print(f"OK — called {tc.name}({tc.arguments})")
            return True
        else:
            print(f"WARN — no tool call, got text: '{response.message.content[:80]}'")
            return False
    except Exception as e:
        print(f"FAIL — {e}")
        return False


async def test_multi_turn(model: OpenAICompatModel) -> bool:
    """测试多轮对话。"""
    print("  [3] Multi-turn...", end=" ", flush=True)
    try:
        messages = [
            Message.user("Remember: my favorite color is blue."),
        ]
        r1 = await model.chat(messages=messages)
        messages.append(r1.message)
        messages.append(Message.user("What is my favorite color?"))
        r2 = await model.chat(messages=messages)
        has_blue = "blue" in (r2.message.content or "").lower()
        print(f"OK — {'remembered' if has_blue else 'forgot'}: '{r2.message.content[:80]}'")
        return has_blue
    except Exception as e:
        print(f"FAIL — {e}")
        return False


async def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY", "sk-cee95412298941d98c15a4bbe0515a5b")
    api_base = os.environ.get("OPENAI_API_BASE", "https://api.deepseek.com/v1")
    model_name = os.environ.get("OPENAI_MODEL_NAME", "deepseek-reasoner")

    print(f"Scaffold Smoke Test")
    print(f"  API base: {api_base}")
    print(f"  Model:    {model_name}")
    print()

    model = OpenAICompatModel(
        api_key=api_key,
        base_url=api_base,
        model=model_name,
    )

    results = []
    results.append(await test_basic_chat(model))
    results.append(await test_tool_call(model))
    results.append(await test_multi_turn(model))

    passed = sum(results)
    total = len(results)
    print(f"\n{'='*40}")
    print(f"  Results: {passed}/{total} passed")
    if passed == total:
        print("  All smoke tests passed! 🎉")
    else:
        print("  Some tests failed — check API key / model compatibility.")
    print(f"{'='*40}")


if __name__ == "__main__":
    asyncio.run(main())
