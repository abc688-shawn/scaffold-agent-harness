"""Skill definition and loader.

A Skill is a self-contained procedural knowledge unit stored as SKILL.md:

    ---
    name: 文件整理
    description: 当用户提到整理/归类文件时触发
    trigger-keywords: [整理, 归类, 按类型, 分组]
    allowed-tools: list_files file_info organize_files tag_files
    metadata:
      version: "1.0"
    ---

    # 正文（Markdown）— 运行时注入 system prompt
    ...

Skills live in an agent-specific directory (e.g., fs_agent/skills/).
The harness only loads and parses them; injection is handled by SkillTriggerMiddleware.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_DELIMITER = "---"


@dataclass
class Skill:
    name: str
    description: str
    trigger_keywords: list[str] = field(default_factory=list)
    allowed_tools: list[str] = field(default_factory=list)
    body: str = ""
    version: str = "1.0"


def _parse_skill_file(path: Path) -> Skill | None:
    text = path.read_text(encoding="utf-8")
    if not text.startswith(_DELIMITER):
        logger.warning("No frontmatter in skill file: %s", path)
        return None

    parts = text.split(_DELIMITER, 2)
    if len(parts) < 3:
        logger.warning("Malformed frontmatter in skill file: %s", path)
        return None

    _, front, body = parts
    try:
        meta = yaml.safe_load(front) or {}
    except yaml.YAMLError as exc:
        logger.warning("YAML error in %s: %s", path, exc)
        return None

    # allowed-tools: space-separated string or list
    raw_tools = meta.get("allowed-tools", "")
    allowed_tools = raw_tools.split() if isinstance(raw_tools, str) else list(raw_tools)

    # trigger-keywords: list or comma-separated string
    raw_kws = meta.get("trigger-keywords", [])
    if isinstance(raw_kws, str):
        trigger_keywords = [k.strip() for k in raw_kws.split(",") if k.strip()]
    else:
        trigger_keywords = list(raw_kws)

    metadata = meta.get("metadata") or {}
    version = str(metadata.get("version", "1.0"))

    return Skill(
        name=meta.get("name", path.parent.name),
        description=meta.get("description", ""),
        trigger_keywords=trigger_keywords,
        allowed_tools=allowed_tools,
        body=body.strip(),
        version=version,
    )


def load_skills(skills_dir: Path) -> list[Skill]:
    """Recursively load all SKILL.md files under *skills_dir*."""
    if not skills_dir.exists():
        return []
    skills = []
    for skill_file in sorted(skills_dir.rglob("SKILL.md")):
        skill = _parse_skill_file(skill_file)
        if skill is not None:
            skills.append(skill)
            logger.debug("Loaded skill '%s' v%s", skill.name, skill.version)
    return skills
