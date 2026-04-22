"""Tests for scaffold/skills — Skill loader and SkillTriggerMiddleware."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from scaffold.skills import Skill, load_skills
from scaffold.skills.loader import _parse_skill_file


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_skill(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "SKILL.md"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# _parse_skill_file
# ---------------------------------------------------------------------------

class TestParseSkillFile:
    def test_full_frontmatter(self, tmp_path):
        path = _write_skill(tmp_path, """\
            ---
            name: 文件整理
            description: 整理文件时触发
            trigger-keywords: [整理, 归类, 分组]
            allowed-tools: list_files move_file
            metadata:
              version: "2.0"
            ---
            # 正文
            整理步骤说明。
        """)
        skill = _parse_skill_file(path)
        assert skill is not None
        assert skill.name == "文件整理"
        assert skill.description == "整理文件时触发"
        assert skill.trigger_keywords == ["整理", "归类", "分组"]
        assert skill.allowed_tools == ["list_files", "move_file"]
        assert skill.version == "2.0"
        assert "整理步骤说明" in skill.body

    def test_allowed_tools_space_separated_string(self, tmp_path):
        path = _write_skill(tmp_path, """\
            ---
            name: 测试
            description: desc
            trigger-keywords: [test]
            allowed-tools: read_file write_file
            ---
            body
        """)
        skill = _parse_skill_file(path)
        assert skill is not None
        assert skill.allowed_tools == ["read_file", "write_file"]

    def test_trigger_keywords_comma_string(self, tmp_path):
        path = _write_skill(tmp_path, """\
            ---
            name: 测试
            description: desc
            trigger-keywords: "对比, 比较, 找差异"
            ---
            body
        """)
        skill = _parse_skill_file(path)
        assert skill is not None
        assert skill.trigger_keywords == ["对比", "比较", "找差异"]

    def test_default_version(self, tmp_path):
        path = _write_skill(tmp_path, """\
            ---
            name: 测试
            description: desc
            ---
            body
        """)
        skill = _parse_skill_file(path)
        assert skill is not None
        assert skill.version == "1.0"

    def test_no_frontmatter_returns_none(self, tmp_path):
        path = tmp_path / "SKILL.md"
        path.write_text("# Just markdown, no frontmatter")
        skill = _parse_skill_file(path)
        assert skill is None

    def test_malformed_yaml_returns_none(self, tmp_path):
        path = _write_skill(tmp_path, """\
            ---
            name: [unclosed bracket
            ---
            body
        """)
        skill = _parse_skill_file(path)
        assert skill is None


# ---------------------------------------------------------------------------
# load_skills
# ---------------------------------------------------------------------------

class TestLoadSkills:
    def test_loads_nested_skills(self, tmp_path):
        for name in ("skill-a", "skill-b"):
            d = tmp_path / name
            d.mkdir()
            (d / "SKILL.md").write_text(
                f"---\nname: {name}\ndescription: d\ntrigger-keywords: [{name}]\n---\nbody\n",
                encoding="utf-8",
            )
        skills = load_skills(tmp_path)
        assert len(skills) == 2
        names = {s.name for s in skills}
        assert "skill-a" in names and "skill-b" in names

    def test_nonexistent_dir_returns_empty(self, tmp_path):
        skills = load_skills(tmp_path / "nonexistent")
        assert skills == []

    def test_skips_invalid_files(self, tmp_path):
        good = tmp_path / "good"
        good.mkdir()
        (good / "SKILL.md").write_text(
            "---\nname: good\ndescription: d\ntrigger-keywords: [good]\n---\nbody\n",
            encoding="utf-8",
        )
        bad = tmp_path / "bad"
        bad.mkdir()
        (bad / "SKILL.md").write_text("no frontmatter here", encoding="utf-8")

        skills = load_skills(tmp_path)
        assert len(skills) == 1
        assert skills[0].name == "good"


# ---------------------------------------------------------------------------
# SkillTriggerMiddleware
# ---------------------------------------------------------------------------

class TestSkillTriggerMiddleware:
    """Tests for the middleware's trigger matching and prompt injection."""

    def _make_ctx(self, user_text: str, tmp_path: Path | None = None):
        """Build a minimal StepContext with a user message."""
        from unittest.mock import MagicMock, AsyncMock
        from scaffold.context.window import AgentPhase, ContextWindow, DynamicPrompt
        from scaffold.loop.middleware import StepContext
        from scaffold.models.base import Message, Role, Usage
        from scaffold.tools.registry import ToolRegistry

        prompt = DynamicPrompt("system base")
        ctx = MagicMock(spec=ContextWindow)
        ctx.prompt = prompt
        ctx.history = [Message(role=Role.USER, content=user_text)]

        return StepContext(
            step=1,
            context=ctx,
            tools=MagicMock(spec=ToolRegistry),
            total_usage=Usage(),
        )

    @pytest.fixture()
    def skill(self) -> Skill:
        return Skill(
            name="文件整理",
            description="整理文件",
            trigger_keywords=["整理", "归类"],
            allowed_tools=["list_files", "move_file"],
            body="## 整理步骤\n1. list_files\n2. move_file",
            version="1.0",
        )

    @pytest.mark.asyncio
    async def test_trigger_injects_section(self, skill):
        from scaffold.context.window import AgentPhase
        from scaffold.loop.middlewares import SkillTriggerMiddleware

        mw = SkillTriggerMiddleware([skill])
        ctx = self._make_ctx("帮我整理一下文件")
        await mw.before_step(ctx)

        assert mw.triggered is skill
        section = ctx.context.prompt.get_phase_prompt(AgentPhase.EXECUTION)
        assert "文件整理" in section
        assert "整理步骤" in section
        assert "list_files" in section  # allowed_tools listed

    @pytest.mark.asyncio
    async def test_no_match_clears_section(self, skill):
        from scaffold.context.window import AgentPhase
        from scaffold.loop.middlewares import SkillTriggerMiddleware

        mw = SkillTriggerMiddleware([skill])
        # Prime a section manually
        ctx = self._make_ctx("请问天气如何")
        ctx.context.prompt.set_phase_prompt(AgentPhase.EXECUTION, "stale section")

        await mw.before_step(ctx)

        assert mw.triggered is None
        section = ctx.context.prompt.get_phase_prompt(AgentPhase.EXECUTION)
        assert section == ""

    @pytest.mark.asyncio
    async def test_step_not_1_is_skipped(self, skill):
        from scaffold.loop.middlewares import SkillTriggerMiddleware
        from scaffold.loop.middleware import StepContext
        from unittest.mock import MagicMock
        from scaffold.context.window import ContextWindow
        from scaffold.tools.registry import ToolRegistry
        from scaffold.models.base import Usage

        mw = SkillTriggerMiddleware([skill])
        ctx = StepContext(
            step=2,
            context=MagicMock(spec=ContextWindow),
            tools=MagicMock(spec=ToolRegistry),
            total_usage=Usage(),
        )
        await mw.before_step(ctx)
        # context.prompt should never be touched
        ctx.context.prompt.set_phase_prompt.assert_not_called()

    @pytest.mark.asyncio
    async def test_case_insensitive_match(self, skill):
        from scaffold.loop.middlewares import SkillTriggerMiddleware

        mw = SkillTriggerMiddleware([skill])
        ctx = self._make_ctx("HELP ME 归类 THESE FILES")
        await mw.before_step(ctx)
        assert mw.triggered is skill

    @pytest.mark.asyncio
    async def test_empty_skills_list(self):
        from scaffold.loop.middlewares import SkillTriggerMiddleware
        from scaffold.loop.middleware import StepContext
        from unittest.mock import MagicMock
        from scaffold.context.window import ContextWindow
        from scaffold.tools.registry import ToolRegistry
        from scaffold.models.base import Usage

        mw = SkillTriggerMiddleware([])
        ctx = StepContext(
            step=1,
            context=MagicMock(spec=ContextWindow),
            tools=MagicMock(spec=ToolRegistry),
            total_usage=Usage(),
        )
        await mw.before_step(ctx)
        ctx.context.prompt.set_phase_prompt.assert_not_called()
