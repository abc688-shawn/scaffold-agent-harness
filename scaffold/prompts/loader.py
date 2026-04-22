"""Jinja2 prompt template loader.

Templates live under scaffold/prompts/ and are referenced by relative path,
e.g. render("system/fs_agent.j2", workspace="/tmp").

build_dynamic_prompt() is a convenience factory that renders the base template
and pre-registers phase-specific sections into a DynamicPrompt object.
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
    """Render a prompt template by relative path, e.g. 'system/fs_agent.j2'."""
    return _get_env().get_template(template_path).render(**kwargs)


def build_dynamic_prompt(base_template: str, **kwargs: Any) -> DynamicPrompt:
    """Render *base_template* and return a DynamicPrompt with phase sections loaded.

    Phase sections are looked up from:
        system/phase_planning.j2
        system/phase_reflection.j2

    Missing phase templates are silently skipped so base-only agents still work.
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
