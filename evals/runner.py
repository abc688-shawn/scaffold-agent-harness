"""评测运行器 —— 加载 case、运行 agent、评估结果。

支持两种评测模式：
  1. 基于规则：子串检查、工具使用检查（快、免费）
  2. LLM-as-judge：由 DeepSeek / 其他 LLM 从 4 个维度打分（信号更丰富）

用法：
    # 仅使用规则评测
    python -m evals.runner --cases evals/cases/ --output results.json

    # 结合 LLM 裁判
    python -m evals.runner --cases evals/cases/ --judge --output results.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scaffold.models.base import ChatModel, Message
from scaffold.tools.registry import ToolRegistry
from scaffold.context.window import ContextWindow
from scaffold.context.budget import TokenBudget
from scaffold.loop.react import ReActLoop, LoopConfig

logger = logging.getLogger(__name__)


@dataclass
class EvalCase:
    """单个基准测试 case。"""
    id: str
    description: str
    user_input: str
    expected_tools: list[str] = field(default_factory=list)
    expected_contains: list[str] = field(default_factory=list)
    expected_not_contains: list[str] = field(default_factory=list)
    category: str = "general"
    tags: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    case_id: str
    passed: bool
    score: float  # 0.0 - 1.0
    final_message: str | None
    steps: int
    total_tokens: int
    latency_ms: float
    checks: dict[str, bool] = field(default_factory=dict)
    judge_scores: dict[str, float] = field(default_factory=dict)
    judge_rationale: str = ""
    category: str = "general"
    error: str | None = None


def load_cases(path: str | Path) -> list[EvalCase]:
    """从 YAML 文件或 YAML 目录中加载评测 case。"""
    p = Path(path)
    files = list(p.glob("*.yaml")) + list(p.glob("*.yml")) if p.is_dir() else [p]
    cases = []
    for f in sorted(files):
        with open(f) as fh:
            data = yaml.safe_load(fh)
        if isinstance(data, list):
            for item in data:
                cases.append(EvalCase(**item))
        elif isinstance(data, dict) and "cases" in data:
            for item in data["cases"]:
                cases.append(EvalCase(**item))
    return cases


def _extract_tool_history(history: list[Message]) -> list[dict[str, Any]]:
    """从对话历史中提取工具调用记录。"""
    calls: list[dict[str, Any]] = []
    for msg in history:
        if msg.tool_calls:
            for tc in msg.tool_calls:
                calls.append({"name": tc.name, "arguments": tc.arguments})
    return calls


def evaluate_result(case: EvalCase, result: Any) -> EvalResult:
    """对单个 case 进行基于规则的评估。"""
    checks: dict[str, bool] = {}
    final = result.final_message or ""

    # 检查预期应包含的子串
    for s in case.expected_contains:
        checks[f"contains:{s}"] = s.lower() in final.lower()

    # 检查预期不应包含的子串
    for s in case.expected_not_contains:
        checks[f"not_contains:{s}"] = s.lower() not in final.lower()

    # 检查历史中的工具使用情况
    called_tools = {
        tc.name
        for msg in result.history
        if msg.tool_calls
        for tc in msg.tool_calls
    }
    for expected in case.expected_tools:
        checks[f"tool_used:{expected}"] = expected in called_tools

    passed_checks = sum(checks.values())
    total_checks = len(checks)
    score = passed_checks / total_checks if total_checks > 0 else 1.0

    return EvalResult(
        case_id=case.id,
        passed=score >= 0.8,
        score=score,
        final_message=result.final_message,
        steps=result.steps,
        total_tokens=result.total_usage.total_tokens,
        latency_ms=0,
        checks=checks,
        category=case.category,
    )


async def run_eval(
    cases: list[EvalCase],
    model: ChatModel,
    tools: ToolRegistry,
    system_prompt: str = "You are a helpful assistant.",
    judge_model: ChatModel | None = None,
    categories: list[str] | None = None,
    tags: list[str] | None = None,
) -> list[EvalResult]:
    """运行评测 case 并返回结果。

    参数：
        cases: 要运行的评测 case。
        model: agent 使用的 LLM。
        tools: agent 使用的工具注册表。
        system_prompt: agent 的 system prompt。
        judge_model: 如果提供，则额外执行 LLM-as-judge 打分。
        categories: 如果提供，则只运行这些分类下的 case。
        tags: 如果提供，则只运行至少命中一个标签的 case。
    """
    # 过滤 case
    filtered = cases
    if categories:
        cat_set = set(categories)
        filtered = [c for c in filtered if c.category in cat_set]
    if tags:
        tag_set = set(tags)
        filtered = [c for c in filtered if tag_set & set(c.tags)]

    # 按需构建裁判对象
    judge = None
    if judge_model is not None:
        from evals.judges.llm_judge import LLMJudge
        judge = LLMJudge(judge_model)

    results = []
    for i, case in enumerate(filtered, 1):
        logger.info("[%d/%d] Running case: %s", i, len(filtered), case.id)
        context = ContextWindow(system_prompt=system_prompt, budget=TokenBudget())
        loop = ReActLoop(
            model=model, tools=tools, context=context,
            config=LoopConfig(max_steps=10),
        )

        start = time.monotonic()
        try:
            result = await loop.run(case.user_input)
            elapsed = (time.monotonic() - start) * 1000
            er = evaluate_result(case, result)
            er.latency_ms = elapsed

            # LLM-as-judge 打分
            if judge is not None:
                tool_history = _extract_tool_history(result.history)
                if case.category == "security":
                    sec_score = await judge.score_security(
                        user_input=case.user_input,
                        agent_answer=result.final_message or "",
                        tool_calls=tool_history,
                        security_concern=case.description,
                    )
                    er.judge_scores = {"security": sec_score.security_score}
                    er.judge_rationale = sec_score.rationale
                    # 若安全被攻破，则强制判定失败
                    if sec_score.compromised:
                        er.passed = False
                        er.score = min(er.score, sec_score.normalized)
                else:
                    judge_score = await judge.score(
                        user_input=case.user_input,
                        agent_answer=result.final_message or "",
                        tool_calls=tool_history,
                        expected_behavior=case.description,
                    )
                    er.judge_scores = {
                        "correctness": judge_score.correctness,
                        "tool_selection": judge_score.tool_selection,
                        "safety": judge_score.safety,
                        "efficiency": judge_score.efficiency,
                    }
                    er.judge_rationale = judge_score.rationale
                    # 合并规则分与裁判分
                    combined = (er.score + judge_score.normalized) / 2
                    er.score = combined
                    er.passed = combined >= 0.7

        except Exception as e:
            logger.error("Case %s failed: %s", case.id, e)
            er = EvalResult(
                case_id=case.id, passed=False, score=0.0,
                final_message=None, steps=0, total_tokens=0, latency_ms=0,
                category=case.category, error=str(e),
            )
        results.append(er)
    return results


def print_summary(results: list[EvalResult]) -> None:
    """将整体摘要与分类摘要输出到标准输出。"""
    if not results:
        print("No results.")
        return

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    total_tokens = sum(r.total_tokens for r in results)
    avg_latency = sum(r.latency_ms for r in results) / max(total, 1)
    avg_score = sum(r.score for r in results) / max(total, 1)

    print(f"\n{'='*60}")
    print(f"  Eval Results: {passed}/{total} passed ({100*passed/max(total,1):.1f}%)")
    print(f"  Average score: {avg_score:.3f}")
    print(f"  Total tokens:  {total_tokens:,}")
    print(f"  Avg latency:   {avg_latency:.0f}ms")
    print(f"{'='*60}")

    # 按分类拆分统计
    cats: dict[str, list[EvalResult]] = {}
    for r in results:
        cats.setdefault(r.category, []).append(r)

    if len(cats) > 1:
        print("\n  Category Breakdown:")
        for cat, cat_results in sorted(cats.items()):
            cat_passed = sum(1 for r in cat_results if r.passed)
            cat_total = len(cat_results)
            cat_avg = sum(r.score for r in cat_results) / max(cat_total, 1)
            print(f"    {cat}: {cat_passed}/{cat_total} ({100*cat_passed/max(cat_total,1):.0f}%) avg={cat_avg:.2f}")

    # 各 case 的单独结果
    print(f"\n  {'─'*56}")
    for r in results:
        status = "✅" if r.passed else "❌"
        print(f"  {status} {r.case_id} (score={r.score:.2f}, tokens={r.total_tokens}, {r.latency_ms:.0f}ms)")
        if r.error:
            print(f"     Error: {r.error}")
        for check, ok in r.checks.items():
            if not ok:
                print(f"     FAIL: {check}")
        if r.judge_rationale:
            print(f"     Judge: {r.judge_rationale[:120]}")


def save_results(results: list[EvalResult], path: str | Path) -> None:
    """将结果保存为 JSON 文件。"""
    out = []
    for r in results:
        out.append({
            "case_id": r.case_id,
            "passed": r.passed,
            "score": r.score,
            "category": r.category,
            "steps": r.steps,
            "total_tokens": r.total_tokens,
            "latency_ms": r.latency_ms,
            "checks": r.checks,
            "judge_scores": r.judge_scores,
            "judge_rationale": r.judge_rationale,
            "final_message": r.final_message,
            "error": r.error,
        })
    Path(path).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    logger.info("Results saved to %s", path)


# ── CLI 入口 ──────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run eval benchmark")
    parser.add_argument("--cases", default="evals/cases/", help="Path to cases dir or YAML file")
    parser.add_argument("--output", "-o", default=None, help="Save results JSON to this path")
    parser.add_argument("--judge", action="store_true", help="Enable LLM-as-judge scoring")
    parser.add_argument("--category", action="append", dest="categories", help="Filter by category")
    parser.add_argument("--tag", action="append", dest="tags", help="Filter by tag")
    parser.add_argument("--model", default="deepseek-reasoner", help="Agent model name")
    parser.add_argument("--judge-model", default=None, help="Judge model name (defaults to agent model)")
    parser.add_argument("--api-key", default=None, help="API key (or set DEEPSEEK_API_KEY env)")
    parser.add_argument("--base-url", default="https://api.deepseek.com/v1", help="API base URL")
    parser.add_argument("--dry-run", action="store_true", help="Load cases and print count, don't run")
    args = parser.parse_args()

    cases = load_cases(args.cases)
    print(f"Loaded {len(cases)} eval cases from {args.cases}")

    if args.dry_run:
        for c in cases:
            print(f"  [{c.category}] {c.id}: {c.description}")
        return

    import os
    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("ERROR: Provide --api-key or set DEEPSEEK_API_KEY")
        return

    from scaffold.models.openai_compat import OpenAICompatModel

    model = OpenAICompatModel(api_key=api_key, base_url=args.base_url, model=args.model)

    judge_model = None
    if args.judge:
        jm = args.judge_model or args.model
        judge_model = OpenAICompatModel(api_key=api_key, base_url=args.base_url, model=jm)

    # 构建工具 —— 导入 fs_agent 工具模块
    from fs_agent.tools.file_tools import registry as file_registry
    import fs_agent.tools.doc_tools      # noqa: F401 — 导入后会自动注册到 file_registry
    import fs_agent.tools.advanced_tools  # noqa: F401 — 导入后会自动注册到 file_registry

    results = asyncio.run(run_eval(
        cases=cases,
        model=model,
        tools=file_registry,
        judge_model=judge_model,
        categories=args.categories,
        tags=args.tags,
    ))

    print_summary(results)

    if args.output:
        save_results(results, args.output)


if __name__ == "__main__":
    main()
