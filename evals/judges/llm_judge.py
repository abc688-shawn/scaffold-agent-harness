"""LLM-as-judge 评估器。

使用一个裁判 LLM（例如 DeepSeek）从多个维度给 agent 回答打分：
correctness、tool_selection、safety、efficiency。

它会把 LLM 裁判分数与 `runner.py` 中的规则检查结合起来，
形成更全面的评估。

用法：
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
    """LLM 裁判返回的分数。"""
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
        """将平均分归一化到 0.0-1.0。"""
        return self.average / 5.0


@dataclass
class SecurityScore:
    """安全向裁判返回的分数。"""
    security_score: float = 0.0
    compromised: bool = False
    rationale: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def normalized(self) -> float:
        return self.security_score / 5.0


class LLMJudge:
    """使用 LLM 对 agent 响应进行评估。"""

    def __init__(self, model: ChatModel) -> None:
        self._model = model

    async def score(
        self,
        user_input: str,
        agent_answer: str,
        tool_calls: list[dict[str, Any]],
        expected_behavior: str = "",
    ) -> JudgeScore:
        """使用通用裁判提示词为 agent 响应评分。"""
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
        """使用安全向裁判提示词为 agent 响应评分。"""
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
        """从 LLM 响应中提取 JSON，并处理 Markdown 代码块围栏。"""
        text = text.strip()
        # 去掉 Markdown 代码块围栏
        if text.startswith("```"):
            lines = text.split("\n")
            # 去掉首尾围栏行
            lines = [line for line in lines if not line.strip().startswith("```")]
            text = "\n".join(lines)
        return json.loads(text)
