"""fs-agent CLI —— 交互式文件助手。

用法：
    fs-agent --workspace ~/Documents --model deepseek-reasoner
    fs-agent --workspace . --api-base https://open.bigmodel.cn/api/paas/v4/ --model glm-5
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env", override=False)
except ImportError:
    pass

from scaffold.cache.cache import ResultCache
from scaffold.context.budget import TokenBudget
from scaffold.context.window import ContextWindow
from scaffold.loop.middlewares import CostTrackerMiddleware, RedactionMiddleware, ToolCallLimitMiddleware
from scaffold.loop.react import LoopConfig, ReActLoop
from scaffold.models.openai_compat import OpenAICompatModel
from scaffold.observability.storage import TraceStorage
from scaffold.observability.tracer import Tracer
from scaffold.prompts.loader import build_dynamic_prompt
from scaffold.safety.injection import INJECTION_DEFENSE_PROMPT
from scaffold.safety.sandbox import PathSandbox

from fs_agent.policies.permissions import FSPermissionGuard, PermissionLevel
from fs_agent.tools.file_tools import registry as file_registry, set_sandbox

# 导入工具模块，使它们自动注册到 file_registry 上
import fs_agent.tools.doc_tools        # noqa: F401 — 导入即完成工具注册
import fs_agent.tools.advanced_tools   # noqa: F401 — 导入即完成工具注册
try:
    import fs_agent.tools.search_tools  # noqa: F401 — 如果依赖可用，则注册语义搜索工具
except ImportError:
    pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="fs-agent: AI file assistant")
    p.add_argument("--workspace", "-w", default=".", help="Root directory for file operations")
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""), help="LLM API key")
    p.add_argument("--api-base", default=os.environ.get("OPENAI_API_BASE", ""), help="LLM API base URL")
    p.add_argument("--model", default=os.environ.get("OPENAI_MODEL_NAME", "deepseek-reasoner"), help="Model name")
    p.add_argument("--max-steps", type=int, default=15, help="Max agent loop steps")
    p.add_argument("--trace-db", default="traces.db", help="SQLite trace database path")
    p.add_argument("--permission", choices=["read_only", "confirm_write", "autonomous"],
                   default="confirm_write", help="Permission level")
    p.add_argument("--embed-model", default=None, help="Embedding model name (enables semantic search)")
    return p


async def run_interactive(args: argparse.Namespace) -> None:
    # --- 初始化 ---
    sandbox = PathSandbox([args.workspace])
    set_sandbox(sandbox)

    level = PermissionLevel(args.permission)
    guard = FSPermissionGuard(level)
    file_registry.set_permission_guard(guard)

    # 启用工具结果缓存（read-only 工具适合缓存，300 s TTL）
    result_cache = ResultCache(default_ttl=300.0)
    file_registry.set_cache(result_cache)

    if not args.api_key:
        print("Error: No API key. Set --api-key or OPENAI_API_KEY env var.", file=sys.stderr)
        sys.exit(1)

    model = OpenAICompatModel(
        api_key=args.api_key,
        base_url=args.api_base or None,
        model=args.model,
    )

    # 可选：配置语义搜索
    try:
        from fs_agent.tools.search_tools import configure_search, EmbeddingClient
        if args.embed_model:
            embed_client = EmbeddingClient(
                api_key=args.api_key,
                base_url=args.api_base or None,
                model=args.embed_model,
            )
            index_path = os.path.join(args.workspace, ".scaffold_index.pkl")
            configure_search(embed_client, index_path)
    except ImportError:
        pass

    # 用 Jinja2 模板构建 DynamicPrompt（包含 PLANNING / REFLECTION 阶段注入）
    dynamic_prompt = build_dynamic_prompt(
        "system/fs_agent.j2",
        workspace=str(sandbox.roots[0]),
        injection_defense=INJECTION_DEFENSE_PROMPT,
    )

    budget = TokenBudget(max_context_tokens=128_000)
    context = ContextWindow(system_prompt=dynamic_prompt, budget=budget)
    trace_storage = TraceStorage(args.trace_db)

    config = LoopConfig(max_steps=args.max_steps)

    # Middleware 栈：循环检测 + 敏感信息脱敏 + Token 成本追踪
    middlewares = [
        ToolCallLimitMiddleware(repeat_limit=3),
        RedactionMiddleware(),
        CostTrackerMiddleware(warn_fraction=0.8, result_cache=result_cache),
    ]

    tools_count = len(file_registry.list_names())
    print(f"🗂  fs-agent ready | workspace: {sandbox.roots[0]} | model: {args.model}")
    print(f"   tools: {tools_count} | permission: {level.value}")
    print("Type your question (or 'quit' to exit):\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        tracer = Tracer()

        loop = ReActLoop(
            model=model,
            tools=file_registry,
            context=context,
            config=config,
            tracer=tracer,
            middlewares=middlewares,
        )

        result = await loop.run(user_input)

        print(f"\nAgent: {result.final_message or '(no response)'}")
        print(f"({result.steps} steps, {result.total_usage.total_tokens} tokens)")
        print()

        trace_storage.save_trace(tracer, metadata={
            "user_input": user_input,
            "steps": result.steps,
        })

    trace_storage.close()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(run_interactive(args))


if __name__ == "__main__":
    main()
