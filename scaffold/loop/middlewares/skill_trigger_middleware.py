"""Skill trigger middleware — detects keywords and injects skill context.

On the first step of every turn, scans the latest user message for trigger
keywords. If a matching skill is found its body is injected into the EXECUTION
phase section of DynamicPrompt, so the model sees structured guidance alongside
the base system prompt without the base prompt being rewritten.

Setting the phase section to "" on each step-1 also clears any skill injected
by a previous turn, so a fresh skill match is performed every time.
"""
from __future__ import annotations

import logging

from scaffold.context.window import AgentPhase
from scaffold.loop.middleware import StepContext, StepMiddleware
from scaffold.models.base import Role
from scaffold.skills.loader import Skill

logger = logging.getLogger(__name__)


class SkillTriggerMiddleware(StepMiddleware):
    """Keyword-based skill injection into the EXECUTION phase prompt.

    Args:
        skills: Pre-loaded Skill objects (from load_skills()).
    """

    def __init__(self, skills: list[Skill]) -> None:
        self._skills = skills
        self._triggered: Skill | None = None

    async def before_step(self, ctx: StepContext) -> None:
        if not self._skills or ctx.step != 1:
            return

        # Clear any skill section from a previous turn.
        ctx.context.prompt.set_phase_prompt(AgentPhase.EXECUTION, "")
        self._triggered = None

        user_text = self._latest_user_message(ctx)
        if not user_text:
            return

        matched = self._find_match(user_text)
        if matched is None:
            return

        self._triggered = matched
        logger.info("Skill '%s' triggered (v%s)", matched.name, matched.version)

        section = f"## 当前激活技能：{matched.name}\n\n{matched.body}"
        if matched.allowed_tools:
            tools_str = ", ".join(f"`{t}`" for t in matched.allowed_tools)
            section += f"\n\n**推荐优先使用的工具**：{tools_str}"

        ctx.context.prompt.set_phase_prompt(AgentPhase.EXECUTION, section)

    @staticmethod
    def _latest_user_message(ctx: StepContext) -> str:
        for msg in reversed(ctx.context.history):
            if msg.role == Role.USER and msg.content:
                return msg.content
        return ""

    def _find_match(self, text: str) -> Skill | None:
        text_lower = text.lower()
        for skill in self._skills:
            if any(kw.lower() in text_lower for kw in skill.trigger_keywords):
                return skill
        return None

    @property
    def triggered(self) -> Skill | None:
        """The skill active during the current turn, or None."""
        return self._triggered
