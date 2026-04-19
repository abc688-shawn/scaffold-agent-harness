"""LLM-as-judge evaluator.

Uses a judge LLM (e.g. DeepSeek) to score agent responses on multiple
dimensions: correctness, tool_selection, safety, efficiency.

Combines LLM judge scores with rule-based checks from runner.py for a
comprehensive evaluation.

Usage:
    judge = LLMJudge(model)
    score = await judge.score(case, agent_answer, tool_history)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from scaffold.models.base import ChatModel, Message

from .prompts import (
    JUDGE_SYSTEM,
    JUDGE_USER_TEMPLATE,
    SECURITY_JUDGE_SYSTEM,
    SECURITY_JUDGE_USER_TEMPLATE,
)

logger = logging.getLogger(__name__)


@dataclass
class JudgeScore:
    """Scores returned by the LLM judge."""
    correctness: float = 0.0
    tool_selection: float = 0.0
    safety: float = 0.0
    efficiency: float = 0.0
    rationale: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def average(self) -> float:
        scores = [self.correctness, self.tool_selection, self.safety, self.efficiency]
        return sum(scores) / len(scores)

    @property
    def normalized(self) -> float:
        """Return average score normalized to 0.0-1.0."""
        return self.average / 5.0


@dataclass
class SecurityScore:
    """Scores returned by the security-focused judge."""
    security_score: float = 0.0
    compromised: bool = False
    rationale: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def normalized(self) -> float:
        return self.security_score / 5.0


class LLMJudge:
    """Uses an LLM to evaluate agent responses."""

    def __init__(self, model: ChatModel) -> None:
        self._model = model

    async def score(
        self,
        user_input: str,
        agent_answer: str,
        tool_calls: list[dict[str, Any]],
        expected_behavior: str = "",
    ) -> JudgeScore:
        """Score an agent response using the general judge prompt."""
        tool_calls_str = self._format_tool_calls(tool_calls)

        user_msg = JUDGE_USER_TEMPLATE.format(
            user_input=user_input,
            agent_answer=agent_answer or "(no answer)",
            tool_calls=tool_calls_str or "(no tools called)",
            expected_behavior=expected_behavior or "(evaluate based on the query)",
        )

        messages = [
            Message.system(JUDGE_SYSTEM),
            Message.user(user_msg),
        ]

        try:
            response = await self._model.chat(messages, temperature=0.0)
            result = self._parse_json(response.message.content or "")
            return JudgeScore(
                correctness=float(result.get("correctness", 0)),
                tool_selection=float(result.get("tool_selection", 0)),
                safety=float(result.get("safety", 0)),
                efficiency=float(result.get("efficiency", 0)),
                rationale=result.get("rationale", ""),
                raw=result,
            )
        except Exception as e:
            logger.error("Judge scoring failed: %s", e)
            return JudgeScore(rationale=f"Judge error: {e}")

    async def score_security(
        self,
        user_input: str,
        agent_answer: str,
        tool_calls: list[dict[str, Any]],
        security_concern: str = "",
    ) -> SecurityScore:
        """Score an agent response using the security-focused judge."""
        tool_calls_str = self._format_tool_calls(tool_calls)

        user_msg = SECURITY_JUDGE_USER_TEMPLATE.format(
            user_input=user_input,
            agent_answer=agent_answer or "(no answer)",
            tool_calls=tool_calls_str or "(no tools called)",
            security_concern=security_concern or "(check for any security issues)",
        )

        messages = [
            Message.system(SECURITY_JUDGE_SYSTEM),
            Message.user(user_msg),
        ]

        try:
            response = await self._model.chat(messages, temperature=0.0)
            result = self._parse_json(response.message.content or "")
            return SecurityScore(
                security_score=float(result.get("security_score", 0)),
                compromised=bool(result.get("compromised", False)),
                rationale=result.get("rationale", ""),
                raw=result,
            )
        except Exception as e:
            logger.error("Security judge scoring failed: %s", e)
            return SecurityScore(rationale=f"Judge error: {e}")

    @staticmethod
    def _format_tool_calls(tool_calls: list[dict[str, Any]]) -> str:
        if not tool_calls:
            return ""
        lines = []
        for tc in tool_calls:
            name = tc.get("name", "unknown")
            args = tc.get("arguments", {})
            lines.append(f"- {name}({json.dumps(args, ensure_ascii=False)})")
        return "\n".join(lines)

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        """Extract JSON from LLM response, handling markdown fences."""
        text = text.strip()
        # Strip markdown code fences
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (fences)
            lines = [line for line in lines if not line.strip().startswith("```")]
            text = "\n".join(lines)
        return json.loads(text)
