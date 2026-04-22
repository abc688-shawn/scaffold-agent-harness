"""Jinja2 提示词模板加载器。

模板存放在 scaffold/prompts/ 下，通过相对路径引用，
例如 render("system/fs_agent.j2", workspace="/tmp")。

build_dynamic_prompt() 是一个便捷工厂函数，用于渲染基础模板
并将各阶段专属片段预注册到 DynamicPrompt 对象中。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import jinja2

from scaffold.context.window import AgentPhase, DynamicPrompt

_PROMPTS_DIR = Path(__file__).parent
_env: jinja2.Environment | None = None


def _get_env() -> jinja2.Environment:
    global _env
    if _env is None:
        _env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(_PROMPTS_DIR)),
            trim_blocks=True,
            lstrip_blocks=True,
            undefined=jinja2.StrictUndefined,
            keep_trailing_newline=True,
        )
    return _env


def render(template_path: str, **kwargs: Any) -> str:
    """通过相对路径渲染提示词模板，例如 'system/fs_agent.j2'。"""
    return _get_env().get_template(template_path).render(**kwargs)


def build_dynamic_prompt(base_template: str, **kwargs: Any) -> DynamicPrompt:
    """渲染 *base_template*，并返回已加载各阶段片段的 DynamicPrompt。

    阶段片段从以下路径查找：
        system/phase_planning.j2
        system/phase_reflection.j2

    缺失的阶段模板会被静默跳过，以兼容仅使用基础提示词的 agent。
    """
    base_text = render(base_template, **kwargs)
    dp = DynamicPrompt(base_text)

    _phase_templates = {
        AgentPhase.PLANNING: "system/phase_planning.j2",
        AgentPhase.REFLECTION: "system/phase_reflection.j2",
    }
    for phase, tmpl_path in _phase_templates.items():
        try:
            section = render(tmpl_path)
            dp.set_phase_prompt(phase, section)
        except jinja2.TemplateNotFound:
            pass

    return dp
