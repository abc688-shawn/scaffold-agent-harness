"""成本追踪 Middleware —— 监控 token 用量与缓存效率。

激活了此前未被调用的 scaffold/cache/cache.py（ResultCache）。
每步都会检查注册表的结果缓存；运行结束时记录命中/未命中统计，
以便评估缓存效果。
"""
from __future__ import annotations

import logging

from scaffold.cache.cache import ResultCache
from scaffold.loop.middleware import StepContext, StepMiddleware

logger = logging.getLogger(__name__)


class CostTrackerMiddleware(StepMiddleware):
    """记录每步的 token 消耗，并在预算不足时发出警告。

    Args:
        warn_fraction: 当 total_tokens / max_total_tokens 超过此比例时触发 WARNING
                       （默认 0.8，即 80%）。
        result_cache:  可选的 ResultCache，用于上报命中/未命中统计。
                       传入与 ToolRegistry 相同的缓存实例即可。
    """

    def __init__(
        self,
        warn_fraction: float = 0.8,
        result_cache: ResultCache | None = None,
    ) -> None:
        self._warn_fraction = warn_fraction
        self._result_cache = result_cache
        self._budget_warned = False

    async def before_step(self, ctx: StepContext) -> None:
        if ctx.max_total_tokens <= 0:
            return
        used = ctx.total_usage.total_tokens
        fraction = used / ctx.max_total_tokens
        if fraction >= self._warn_fraction and not self._budget_warned:
            self._budget_warned = True
            logger.warning(
                "Token budget %.0f%% used — %d / %d tokens",
                fraction * 100,
                used,
                ctx.max_total_tokens,
            )

    async def after_step(self, ctx: StepContext) -> None:
        logger.debug(
            "Step %d — tokens: %d total (prompt=%d, completion=%d)",
            ctx.step,
            ctx.total_usage.total_tokens,
            ctx.total_usage.prompt_tokens,
            ctx.total_usage.completion_tokens,
        )
        if self._result_cache is not None:
            stats = self._result_cache.stats
            logger.debug(
                "ResultCache — hits=%d misses=%d size=%d",
                stats["hits"],
                stats["misses"],
                stats["size"],
            )
