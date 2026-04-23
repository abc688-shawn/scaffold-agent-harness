"""MCP Client 适配层 — 将外部 MCP Server 的工具代理进 Scaffold ToolRegistry。

使用 Anthropic 官方维护的 mcp-server-fetch（无需编写服务端代码，uvx 直接运行）：
  https://github.com/modelcontextprotocol/servers/tree/main/src/fetch

架构：
  LLM → Function Calling → Scaffold ToolRegistry
                                  ↓
                          _call_mcp_fetch()
                                  ↓
                  uvx mcp-server-fetch（独立子进程，stdio）
                                  ↓
                          fetch(url=...)  →  wttr.in 天气 API
"""
from __future__ import annotations

from scaffold.tools.errors import ToolError, ToolErrorCode
from fs_agent.tools.file_tools import registry

# mcp-server-fetch 通过 uvx 启动，无需本地安装
_FETCH_SERVER = {"command": "uvx", "args": ["mcp-server-fetch"]}


async def _call_mcp_fetch(url: str) -> str:
    """通过 mcp-server-fetch 抓取指定 URL，返回文本内容。"""
    try:
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client
    except ImportError:
        raise ToolError(
            ToolErrorCode.INTERNAL_ERROR,
            "mcp package required. Install with: uv sync --extra mcp",
        )

    params = StdioServerParameters(
        command=_FETCH_SERVER["command"],
        args=_FETCH_SERVER["args"],
    )
    try:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("fetch", {"url": url})
                texts = [c.text for c in result.content if hasattr(c, "text")]
                return "\n".join(texts) if texts else "(no response)"
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(ToolErrorCode.INTERNAL_ERROR, f"MCP fetch failed: {e}")


# ---------------------------------------------------------------------------
# 工具：get_current_weather
# ---------------------------------------------------------------------------

@registry.tool(
    name="get_current_weather",
    description=(
        "获取指定城市的当前天气，包括温度和天气状况。"
        "当用户询问天气、温度、下雨、晴天等问题时使用。"
        "city 参数支持中英文，如 Beijing、上海、Tokyo。"
    ),
)
async def get_current_weather(city: str) -> str:
    """查询城市当前天气。

    city: 城市名称（支持中英文，如 Beijing、上海）。
    """
    # wttr.in 的 format=3 返回一行简洁天气，如：Beijing: ☀️ +25°C
    url = f"https://wttr.in/{city}?format=3"
    return await _call_mcp_fetch(url)


# ---------------------------------------------------------------------------
# 工具：get_weather_forecast
# ---------------------------------------------------------------------------

@registry.tool(
    name="get_weather_forecast",
    description=(
        "获取指定城市未来 3 天的天气预报。"
        "当用户询问明天、后天或未来几天天气时使用。"
    ),
)
async def get_weather_forecast(city: str) -> str:
    """查询城市 3 天天气预报。

    city: 城市名称（支持中英文）。
    """
    # format=4 返回三行，每行一天的预报
    url = f"https://wttr.in/{city}?format=4"
    return await _call_mcp_fetch(url)
