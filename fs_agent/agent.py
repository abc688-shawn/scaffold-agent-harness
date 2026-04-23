"""FSAgent —— 文件管理 Agent 的统一入口。

替代了 cli.py、app.py 和 evals/runner.py 中重复的启动代码。
以上三个入口应统一通过实例化 FSAgent 并调用 run() 来使用。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from scaffold.cache.cache import ResultCache
from scaffold.context.budget import TokenBudget
from scaffold.context.window import ContextWindow
from scaffold.loop.checkpoint import CheckpointRecord, CheckpointStore
from scaffold.loop.middleware import StepMiddleware
from scaffold.loop.middlewares import CostTrackerMiddleware, RedactionMiddleware, ToolCallLimitMiddleware
from scaffold.loop.middlewares.skill_trigger_middleware import SkillTriggerMiddleware
from scaffold.loop.react import LoopConfig, LoopResult, ReActLoop
from scaffold.models.base import Usage
from scaffold.models.base import ChatModel
from scaffold.models.openai_compat import OpenAICompatModel
from scaffold.observability.tracer import Tracer
from scaffold.prompts.loader import build_dynamic_prompt
from scaffold.safety.injection import INJECTION_DEFENSE_PROMPT
from scaffold.safety.sandbox import PathSandbox
from scaffold.skills import load_skills

from fs_agent.policies.permissions import FSPermissionGuard, PermissionLevel
from fs_agent.tools.file_tools import registry, set_sandbox

import fs_agent.tools.doc_tools        # noqa: F401 — registers on import
import fs_agent.tools.advanced_tools   # noqa: F401 — registers on import
import fs_agent.tools.reference_tools  # noqa: F401 — registers retrieve_reference tool
import fs_agent.tools.skill_tools      # noqa: F401 — registers list_skills tool
try:
    import fs_agent.tools.search_tools  # noqa: F401
except ImportError:
    pass
try:
    import fs_agent.tools.mcp_tools     # noqa: F401 — registers MCP-proxied tools
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
    checkpoint_db: str | None = "traces.db"
    embedding_api_key: str = ""
    embedding_api_base: str = ""
    embedding_model: str = ""


class FSAgent:
    """独立的文件管理 Agent。

    实例化一次后可多次调用 run()。每次 run() 都会创建新的 Middleware 栈，
    确保每轮的工具调用计数器从零开始。跨轮次保持对话历史时，
    请将同一个 ContextWindow 传给每次 run() 调用。
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

        from fs_agent.tools.skill_tools import set_skills
        set_skills(self._skills)

        self._configure_embedding(config)

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def new_context(self) -> ContextWindow:
        """返回一个为该 Agent 配置好的新 ContextWindow。"""
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
        """执行一轮 Agent 对话。

        Args:
            user_input: 本轮用户的输入消息。
            context:    要使用的 ContextWindow。若为 None 则创建新实例
                        （无状态/单轮模式）。多轮对话时请复用同一实例以保留历史。
            tracer:     可观测性追踪器（可选）。
        """
        ctx = context if context is not None else self.new_context()
        self._wire_ref_store(ctx)
        tracer = tracer or Tracer()
        middlewares = self._build_middlewares()

        checkpoint_store, run_id = self._open_checkpoint_store()
        try:
            loop = ReActLoop(
                model=self._model,
                tools=registry,
                context=ctx,
                config=self._loop_config,
                tracer=tracer,
                middlewares=middlewares,
                checkpoint_store=checkpoint_store,
                run_id=run_id,
            )
            return await loop.run(user_input)
        finally:
            if checkpoint_store:
                checkpoint_store.close()

    async def resume(
        self,
        run_id: str,
        *,
        checkpoint_db: str | None = None,
        tracer: Tracer | None = None,
    ) -> LoopResult:
        """从已保存的检查点恢复中断的运行。

        Args:
            run_id:        要恢复的运行 ID（来自 list_incomplete_runs()）。
            checkpoint_db: 检查点数据库路径，默认使用 config.checkpoint_db。
            tracer:        可观测性追踪器（可选）。

        Raises:
            ValueError: 如果 run_id 不存在或已完成。
        """
        db = checkpoint_db or self._config.checkpoint_db or "traces.db"
        store = CheckpointStore(db)
        record = store.load(run_id)

        if record is None:
            store.close()
            raise ValueError(f"Checkpoint '{run_id}' not found in {db!r}")
        if record.completed:
            store.close()
            raise ValueError(f"Run '{run_id}' is already completed")

        logger.info("Resuming run '%s' from step %d", run_id, record.step)

        ctx = self.new_context()
        for msg in record.messages:
            ctx.add(msg)
        self._wire_ref_store(ctx)

        # Restore accumulated usage so budget tracking stays accurate
        loop = ReActLoop(
            model=self._model,
            tools=registry,
            context=ctx,
            config=self._loop_config,
            tracer=tracer or Tracer(),
            middlewares=self._build_middlewares(),
            checkpoint_store=store,
            run_id=run_id,
        )
        loop._total_usage = record.usage

        try:
            return await loop.run(record.user_input, _resume_step=record.step)
        finally:
            store.close()

    def list_incomplete_runs(self, *, checkpoint_db: str | None = None, limit: int = 10) -> list[CheckpointRecord]:
        """返回尚未完成的运行列表（按最新更新时间降序）。"""
        db = checkpoint_db or self._config.checkpoint_db or "traces.db"
        store = CheckpointStore(db)
        try:
            return store.list_incomplete(limit=limit)
        finally:
            store.close()

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    def _configure_embedding(self, config: FSAgentConfig) -> None:
        import os
        try:
            from fs_agent.tools.search_tools import configure_search, EmbeddingClient
        except ImportError:
            return
        key = config.embedding_api_key or os.environ.get("EMBEDDING_API_KEY", "")
        base = config.embedding_api_base or os.environ.get("EMBEDDING_API_BASE", "")
        model = config.embedding_model or os.environ.get("EMBEDDING_MODEL_NAME", "")
        if key and model:
            configure_search(EmbeddingClient(api_key=key, base_url=base or None, model=model))
            logger.info("Embedding client configured: model=%s", model)

    def _wire_ref_store(self, ctx: ContextWindow) -> None:
        from fs_agent.tools.reference_tools import set_ref_store
        set_ref_store(ctx.ref_store)

    def _open_checkpoint_store(self) -> tuple[CheckpointStore | None, str | None]:
        if not self._config.checkpoint_db:
            return None, None
        store = CheckpointStore(self._config.checkpoint_db)
        run_id = CheckpointStore.new_run_id()
        return store, run_id

    def _build_middlewares(self) -> list[StepMiddleware]:
        ms: list[StepMiddleware] = [
            ToolCallLimitMiddleware(repeat_limit=3),
            RedactionMiddleware(),
            CostTrackerMiddleware(warn_fraction=0.8, result_cache=self._result_cache),
        ]
        if self._skills:
            ms.append(SkillTriggerMiddleware(self._skills))
        return ms
