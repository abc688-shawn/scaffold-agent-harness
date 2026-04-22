"""Cost-tracker middleware — monitors token usage and cache efficiency.

Activates scaffold/cache/cache.py (ResultCache) which was previously unreachable.
The registry's result cache is checked each step; hit/miss stats are logged at
the end of the run to give a feedback loop on cache effectiveness.
"""
from __future__ import annotations

import logging

from scaffold.cache.cache import ResultCache
from scaffold.loop.middleware import StepContext, StepMiddleware

logger = logging.getLogger(__name__)


class CostTrackerMiddleware(StepMiddleware):
    """Log token spend per step and warn when the budget is running low.

    Args:
        warn_fraction: Emit a WARNING when total_tokens / max_total_tokens
                       exceeds this fraction (default 0.8 = 80 %).
        result_cache:  Optional ResultCache to report hit/miss stats.
                       Pass the same cache instance you gave to ToolRegistry.
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
