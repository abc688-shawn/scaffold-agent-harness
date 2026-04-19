"""fs-agent CLI — interactive file assistant.

Usage:
    fs-agent --workspace ~/Documents --model deepseek-reasoner
    fs-agent --workspace . --api-base https://open.bigmodel.cn/api/paas/v4/ --model glm-5
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

from scaffold.models.openai_compat import OpenAICompatModel
from scaffold.context.window import ContextWindow
from scaffold.context.budget import TokenBudget
from scaffold.loop.react import ReActLoop, LoopConfig
from scaffold.observability.tracer import Tracer
from scaffold.observability.storage import TraceStorage
from scaffold.safety.sandbox import PathSandbox
from scaffold.safety.injection import INJECTION_DEFENSE_PROMPT

from fs_agent.tools.file_tools import registry as file_registry, set_sandbox
from fs_agent.policies.permissions import PermissionLevel, FSPermissionGuard

# Import tool modules so they register onto file_registry
import fs_agent.tools.doc_tools        # noqa: F401
import fs_agent.tools.advanced_tools   # noqa: F401
try:
    import fs_agent.tools.search_tools  # noqa: F401 — registers semantic search tools if deps available
except ImportError:
    pass


SYSTEM_PROMPT = """You are a helpful file system assistant. You can browse, read, search, and manage files in the user's workspace.

When the user asks a question:
1. First understand what they need
2. Use the available tools to find information
3. Provide a clear, concise answer based on what you found

Always tell the user what you found and cite specific files when relevant.

{injection_defense}
""".format(injection_defense=INJECTION_DEFENSE_PROMPT)


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
    # --- Setup ---
    sandbox = PathSandbox([args.workspace])
    set_sandbox(sandbox)

    # Permission guard
    level = PermissionLevel(args.permission)
    guard = FSPermissionGuard(level)
    file_registry.set_permission_guard(guard)

    if not args.api_key:
        print("Error: No API key. Set --api-key or OPENAI_API_KEY env var.", file=sys.stderr)
        sys.exit(1)

    model = OpenAICompatModel(
        api_key=args.api_key,
        base_url=args.api_base or None,
        model=args.model,
    )

    # Optional: configure semantic search
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

    budget = TokenBudget(max_context_tokens=128_000)
    context = ContextWindow(system_prompt=SYSTEM_PROMPT, budget=budget)
    trace_storage = TraceStorage(args.trace_db)

    config = LoopConfig(max_steps=args.max_steps)

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
        )

        result = await loop.run(user_input)

        print(f"\nAgent ({result.steps} steps, {result.total_usage.total_tokens} tokens):")
        print(result.final_message or "(no response)")
        print()

        # Save trace
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
