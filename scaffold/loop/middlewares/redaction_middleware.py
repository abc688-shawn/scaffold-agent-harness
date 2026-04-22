"""Redaction middleware — strips sensitive data from tool results.

Activates scaffold/safety/redaction.py which was previously unreachable dead code.
Runs after every tool call and redacts API keys, passwords, emails, phone numbers,
and Chinese ID numbers before the result is added to the conversation context.
"""
from __future__ import annotations

import logging

from scaffold.loop.middleware import StepContext, StepMiddleware
from scaffold.models.base import ToolCall, ToolResult
from scaffold.safety.redaction import redact_sensitive

logger = logging.getLogger(__name__)


class RedactionMiddleware(StepMiddleware):
    """Redact PII / secrets from tool results before they enter the context."""

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
