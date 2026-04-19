"""Tests for dynamic system prompts and phase-aware context."""
from __future__ import annotations

import pytest

from scaffold.context.window import AgentPhase, DynamicPrompt, ContextWindow
from scaffold.context.budget import TokenBudget
from scaffold.models.base import Message


class TestDynamicPrompt:
    def test_basic_render(self):
        dp = DynamicPrompt("You are a helpful assistant.")
        assert dp.render() == "You are a helpful assistant."

    def test_phase_prompt_rendering(self):
        dp = DynamicPrompt("Base prompt.")
        dp.set_phase_prompt(AgentPhase.PLANNING, "Focus on planning steps.")
        dp.set_phase_prompt(AgentPhase.EXECUTION, "Execute the plan.")
        dp.set_phase_prompt(AgentPhase.REFLECTION, "Reflect on what happened.")

        # Default phase is EXECUTION
        assert dp.phase == AgentPhase.EXECUTION
        rendered = dp.render()
        assert "Base prompt." in rendered
        assert "Execute the plan." in rendered
        assert "planning" not in rendered.lower().split("execution")[0]  # planning text not present

    def test_phase_switching(self):
        dp = DynamicPrompt("Base.")
        dp.set_phase_prompt(AgentPhase.PLANNING, "Plan section.")
        dp.set_phase_prompt(AgentPhase.EXECUTION, "Exec section.")

        dp.phase = AgentPhase.PLANNING
        assert "Plan section." in dp.render()
        assert "Exec section." not in dp.render()

        dp.phase = AgentPhase.EXECUTION
        assert "Exec section." in dp.render()
        assert "Plan section." not in dp.render()

    def test_kv_cache_friendly_layout(self):
        """Base prompt should always be the prefix (for KV-cache)."""
        dp = DynamicPrompt("STABLE_PREFIX_CONTENT")
        dp.set_phase_prompt(AgentPhase.PLANNING, "dynamic planning content")

        dp.phase = AgentPhase.PLANNING
        rendered = dp.render()
        assert rendered.startswith("STABLE_PREFIX_CONTENT")

    def test_no_phase_section(self):
        """Phase without registered section falls back to base only."""
        dp = DynamicPrompt("Base only.")
        dp.phase = AgentPhase.REFLECTION
        assert dp.render() == "Base only."


class TestContextWindowWithDynamicPrompt:
    def test_string_prompt_creates_dynamic(self):
        cw = ContextWindow(system_prompt="Simple string")
        assert isinstance(cw.prompt, DynamicPrompt)
        assert cw.prompt.render() == "Simple string"

    def test_dynamic_prompt_passthrough(self):
        dp = DynamicPrompt("Custom prompt")
        dp.set_phase_prompt(AgentPhase.PLANNING, "Planning mode")
        cw = ContextWindow(system_prompt=dp)
        assert cw.prompt is dp

    def test_set_phase_via_context(self):
        dp = DynamicPrompt("Base")
        dp.set_phase_prompt(AgentPhase.PLANNING, "Planning")
        cw = ContextWindow(system_prompt=dp)

        cw.prompt.phase = AgentPhase.PLANNING
        prompt = cw.build_prompt()
        # System message should contain planning content
        system_msg = prompt[0]
        assert "Planning" in system_msg.content

    def test_build_prompt_includes_history(self):
        cw = ContextWindow(system_prompt="System")
        cw.add(Message.user("Hello"))
        cw.add(Message.assistant("Hi there"))

        prompt = cw.build_prompt()
        assert len(prompt) >= 3  # system + user + assistant
        assert prompt[0].content == "System"
