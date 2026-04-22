"""技能触发 Middleware —— 检测关键词并注入技能上下文。

在每轮的第一步扫描最新的用户消息，检查是否命中触发关键词。
若找到匹配技能，则将其正文注入 DynamicPrompt 的 EXECUTION 阶段片段，
使模型在不重写基础 system prompt 的前提下看到结构化指导。

每次第 1 步时将阶段片段重置为 ""，以清除上一轮注入的技能，
确保每轮都重新匹配。
"""
from __future__ import annotations

import logging

from scaffold.context.window import AgentPhase
from scaffold.loop.middleware import StepContext, StepMiddleware
from scaffold.models.base import Role
from scaffold.skills.loader import Skill

logger = logging.getLogger(__name__)


class SkillTriggerMiddleware(StepMiddleware):
    """基于关键词匹配，将技能注入 EXECUTION 阶段提示词。

    Args:
        skills: 已预加载的 Skill 对象列表（来自 load_skills()）。
    """

    def __init__(self, skills: list[Skill]) -> None:
        self._skills = skills
        self._triggered: Skill | None = None

    async def before_step(self, ctx: StepContext) -> None:
        if not self._skills or ctx.step != 1:
            return

        # 清除上一轮注入的技能片段。
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
        """当前轮次激活的技能，若未命中则为 None。"""
        return self._triggered
