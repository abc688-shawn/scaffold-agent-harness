"""list_skills 工具 — 列出当前 Agent 加载的所有技能。

用法：用户可以向 Agent 提问"我现在有哪些技能？"，
Agent 调用此工具即可获取完整的技能列表。
"""
from __future__ import annotations

from scaffold.skills.loader import Skill
from fs_agent.tools.file_tools import registry

_skills: list[Skill] = []


def set_skills(skills: list[Skill]) -> None:
    """将 FSAgent 加载的 skill 列表注入本模块。"""
    global _skills
    _skills = skills


@registry.tool(
    name="list_skills",
    description=(
        "List all skills currently available to this agent. "
        "Call this when the user asks what skills, capabilities, or special modes are available. "
        "Returns each skill's name, description, trigger keywords, and recommended tools."
    ),
)
async def list_skills() -> str:
    """返回当前加载的所有技能的详细信息。"""
    if not _skills:
        return "当前没有加载任何技能。"

    lines: list[str] = [f"当前共加载了 {len(_skills)} 个技能：\n"]
    for i, skill in enumerate(_skills, 1):
        lines.append(f"## {i}. {skill.name}（v{skill.version}）")
        lines.append(f"**描述**：{skill.description}")
        if skill.trigger_keywords:
            kws = "、".join(skill.trigger_keywords)
            lines.append(f"**触发关键词**：{kws}")
        if skill.allowed_tools:
            tools = "、".join(skill.allowed_tools)
            lines.append(f"**推荐工具**：{tools}")
        lines.append("")

    return "\n".join(lines)
