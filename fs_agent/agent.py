"""FSAgent — unified entry point for the file-management agent.

Replaces the duplicated setup code in cli.py, app.py and evals/runner.py.
All three entry points should instantiate FSAgent and call run().
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from scaffold.cache.cache import ResultCache
from scaffold.context.budget import TokenBudget
from scaffold.context.window import ContextWindow
from scaffold.loop.middleware import StepMiddleware
from scaffold.loop.middlewares import CostTrackerMiddleware, RedactionMiddleware, ToolCallLimitMiddleware
from scaffold.loop.middlewares.skill_trigger_middleware import SkillTriggerMiddleware
from scaffold.loop.react import LoopConfig, LoopResult, ReActLoop
from scaffold.models.base import ChatModel
from scaffold.models.openai_compat import OpenAICompatModel
from scaffold.observability.storage import TraceStorage
from scaffold.observability.tracer import Tracer
from scaffold.prompts.loader import build_dynamic_prompt
from scaffold.safety.injection import INJECTION_DEFENSE_PROMPT
from scaffold.safety.sandbox import PathSandbox
from scaffold.skills import load_skills

from fs_agent.policies.permissions import FSPermissionGuard, PermissionLevel
from fs_agent.tools.file_tools import registry, set_sandbox

import fs_agent.tools.doc_tools        # noqa: F401 — registers on import
import fs_agent.tools.advanced_tools   # noqa: F401 — registers on import
try:
    import fs_agent.tools.search_tools  # noqa: F401
except ImportError:
    pass

logger = logging.getLogger(__name__)

_SKILLS_DIR = Path(__file__).parent / "skills"


@dataclass
class FSAgentConfig:
    workspace: str | Path
    api_key: str = ""
    api_base: str = ""
    model: str = "deepseek-reasoner"
    max_steps: int = 15
    max_context_tokens: int = 128_000
    permission: str = "confirm_write"
    cache_ttl: float = 300.0
    skills_dir: Path | None = None


class FSAgent:
    """Self-contained file-management agent.

    Instantiate once and call run() repeatedly.  Each run() creates a fresh
    middleware stack so per-turn call counters start clean.  Pass the same
    ContextWindow across calls to maintain conversational history.
    """

    def __init__(
        self,
        config: FSAgentConfig,
        *,
        model: ChatModel | None = None,
    ) -> None:
        self._config = config
        workspace = str(Path(config.workspace).expanduser().resolve())

        sandbox = PathSandbox([workspace])
        set_sandbox(sandbox)

        level = PermissionLevel(config.permission)
        registry.set_permission_guard(FSPermissionGuard(level))

        self._result_cache = ResultCache(default_ttl=config.cache_ttl)
        registry.set_cache(self._result_cache)

        self._model: ChatModel = model or OpenAICompatModel(
            api_key=config.api_key,
            base_url=config.api_base or None,
            model=config.model,
        )

        self._dynamic_prompt = build_dynamic_prompt(
            "system/fs_agent.j2",
            workspace=workspace,
            injection_defense=INJECTION_DEFENSE_PROMPT,
        )

        self._budget = TokenBudget(max_context_tokens=config.max_context_tokens)
        self._loop_config = LoopConfig(max_steps=config.max_steps)

        skills_dir = config.skills_dir or _SKILLS_DIR
        self._skills = load_skills(skills_dir)
        if self._skills:
            logger.info("Loaded %d skill(s) from %s", len(self._skills), skills_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def new_context(self) -> ContextWindow:
        """Return a fresh ContextWindow configured for this agent."""
        return ContextWindow(system_prompt=self._dynamic_prompt, budget=self._budget)

    @property
    def result_cache(self) -> ResultCache:
        return self._result_cache

    async def run(
        self,
        user_input: str,
        *,
        context: ContextWindow | None = None,
        tracer: Tracer | None = None,
    ) -> LoopResult:
        """Run one agent turn.

        Args:
            user_input: The user's message for this turn.
            context: ContextWindow to use.  If None a fresh one is created
                     (stateless / single-turn mode).  Pass the same instance
                     across calls to keep conversational history (stateful mode).
            tracer: Optional tracer for observability.
        """
        ctx = context if context is not None else self.new_context()
        tracer = tracer or Tracer()
        middlewares = self._build_middlewares()

        loop = ReActLoop(
            model=self._model,
            tools=registry,
            context=ctx,
            config=self._loop_config,
            tracer=tracer,
            middlewares=middlewares,
        )
        return await loop.run(user_input)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_middlewares(self) -> list[StepMiddleware]:
        ms: list[StepMiddleware] = [
            ToolCallLimitMiddleware(repeat_limit=3),
            RedactionMiddleware(),
            CostTrackerMiddleware(warn_fraction=0.8, result_cache=self._result_cache),
        ]
        if self._skills:
            ms.append(SkillTriggerMiddleware(self._skills))
        return ms
