"""脱敏 Middleware —— 从工具结果中移除敏感数据。

激活了此前未被调用的死代码 scaffold/safety/redaction.py。
在每次工具调用后执行，将结果写入对话上下文之前，
自动脱敏 API Key、密码、邮箱、手机号和中国身份证号。
"""
from __future__ import annotations

import logging

from scaffold.loop.middleware import StepContext, StepMiddleware
from scaffold.models.base import ToolCall, ToolResult
from scaffold.safety.redaction import redact_sensitive

logger = logging.getLogger(__name__)


class RedactionMiddleware(StepMiddleware):
    """在工具结果写入上下文之前，对其中的 PII / 密钥进行脱敏处理。"""

    async def after_tool(
        self, ctx: StepContext, call: ToolCall, result: ToolResult
    ) -> ToolResult:
        if not result.content:
            return result

        redacted = redact_sensitive(result.content)
        if redacted is result.content:
            return result

        logger.debug("Redacted sensitive data in result of '%s'", call.name)
        return ToolResult(
            tool_call_id=result.tool_call_id,
            name=result.name,
            content=redacted,
            is_error=result.is_error,
        )
