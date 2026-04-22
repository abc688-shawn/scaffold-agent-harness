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

from scaffold.observability.storage import TraceStorage
from scaffold.observability.tracer import Tracer

from fs_agent.agent import FSAgent, FSAgentConfig


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
    return p


async def run_interactive(args: argparse.Namespace) -> None:
    if not args.api_key:
        print("Error: No API key. Set --api-key or OPENAI_API_KEY env var.", file=sys.stderr)
        sys.exit(1)

    config = FSAgentConfig(
        workspace=args.workspace,
        api_key=args.api_key,
        api_base=args.api_base,
        model=args.model,
        max_steps=args.max_steps,
        permission=args.permission,
    )
    agent = FSAgent(config)
    context = agent.new_context()   # shared across turns → accumulates history
    trace_storage = TraceStorage(args.trace_db)

    from fs_agent.tools.file_tools import registry
    tools_count = len(registry.list_names())
    print(f"🗂  fs-agent ready | workspace: {args.workspace} | model: {args.model}")
    print(f"   tools: {tools_count} | permission: {args.permission}")
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
        result = await agent.run(user_input, context=context, tracer=tracer)

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
