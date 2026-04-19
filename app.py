"""Scaffold fs-agent — interactive web chat UI.

Launch:
    streamlit run app.py

Features:
- Upload documents (PDF, DOCX, TXT, CSV, etc.) to a persistent local library
- Chat with the fs-agent about your files
- Sidebar shows API config, document library, and per-turn token stats
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Load .env so API credentials don't need to be set manually each time
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env", override=False)
except ImportError:
    pass

try:
    import streamlit as st
except ImportError:
    print("Install streamlit: pip install streamlit", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Scaffold fs-agent",
    page_icon="🗂",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Default persistent document library — lives inside the project directory
_DEFAULT_WORKSPACE = Path(__file__).parent / "workspace"

# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def _init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []   # list of {"role", "content", "meta"}
    if "total_tokens" not in st.session_state:
        st.session_state.total_tokens = 0
    if "total_steps" not in st.session_state:
        st.session_state.total_steps = 0

_init_state()

# ---------------------------------------------------------------------------
# Sidebar — config & file upload
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🗂 fs-agent")
    st.caption("Scaffold · Agent Harness Demo")
    st.divider()

    # ── API Configuration ────────────────────────────────────────────────
    st.subheader("API 配置")
    api_key = st.text_input(
        "API Key",
        value=os.environ.get("OPENAI_API_KEY", ""),
        type="password",
        placeholder="sk-...",
    )
    api_base = st.text_input(
        "API Base URL",
        value=os.environ.get("OPENAI_API_BASE", "https://open.bigmodel.cn/api/paas/v4/"),
        placeholder="https://open.bigmodel.cn/api/paas/v4/",
    )
    model_name = st.text_input(
        "Model",
        value=os.environ.get("OPENAI_MODEL_NAME", "glm-5"),
        placeholder="glm-5",
    )
    permission = st.selectbox(
        "权限级别",
        ["confirm_write", "read_only", "autonomous"],
        index=0,
        help="read_only: 只读 | confirm_write: 写操作需确认 | autonomous: 完全自主",
    )
    max_steps = st.slider("最大步数", min_value=5, max_value=30, value=15)

    st.divider()

    # ── Document Library ─────────────────────────────────────────────────
    st.subheader("文档库")

    workspace_input = st.text_input(
        "库目录路径",
        value=str(_DEFAULT_WORKSPACE),
        help="文件永久保存在此目录，重启应用后仍保留",
    )
    workspace_path = Path(workspace_input).expanduser().resolve()
    workspace_path.mkdir(parents=True, exist_ok=True)

    uploaded = st.file_uploader(
        "上传文档（PDF、DOCX、TXT、CSV、Markdown 等）",
        accept_multiple_files=True,
        type=["pdf", "docx", "txt", "csv", "json", "md", "yaml", "yml", "py", "js", "ts"],
    )

    # Save uploaded files to the persistent workspace
    if uploaded:
        for uf in uploaded:
            dest = workspace_path / uf.name
            dest.write_bytes(uf.read())   # overwrite = update existing file
            st.toast(f"已保存: {uf.name}", icon="✅")

    # List all files in library
    all_files = sorted(
        f for f in workspace_path.rglob("*") if f.is_file() and not f.name.startswith(".")
    )
    if all_files:
        st.caption(f"{len(all_files)} 个文件 · `{workspace_path}`")
        for f in all_files:
            size = f.stat().st_size
            if size >= 1024 * 1024:
                size_str = f"{size/1024/1024:.1f}MB"
            elif size >= 1024:
                size_str = f"{size/1024:.1f}KB"
            else:
                size_str = f"{size}B"
            rel = f.relative_to(workspace_path)
            st.text(f"  📄 {rel}  ({size_str})")
    else:
        st.info("文档库为空，请上传文档后再提问。")

    st.divider()

    # ── Stats ────────────────────────────────────────────────────────────
    st.subheader("会话统计")
    col1, col2 = st.columns(2)
    col1.metric("总 Tokens", f"{st.session_state.total_tokens:,}")
    col2.metric("总步数", st.session_state.total_steps)

    if st.button("🗑 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_tokens = 0
        st.session_state.total_steps = 0
        st.rerun()

# ---------------------------------------------------------------------------
# Agent builder (cached per config)
# ---------------------------------------------------------------------------

@st.cache_resource(hash_funcs={"builtins.str": lambda s: s})
def _build_agent_components(api_key: str, api_base: str, model: str,
                             workspace: str, permission_level: str, max_steps: int):
    """Build and return the agent components. Cached until config changes."""
    from scaffold.models.openai_compat import OpenAICompatModel
    from scaffold.context.window import ContextWindow
    from scaffold.context.budget import TokenBudget
    from scaffold.loop.react import LoopConfig
    from scaffold.safety.sandbox import PathSandbox
    from scaffold.safety.injection import INJECTION_DEFENSE_PROMPT
    from fs_agent.tools.file_tools import registry, set_sandbox
    from fs_agent.policies.permissions import PermissionLevel, FSPermissionGuard
    import fs_agent.tools.doc_tools       # noqa: F401
    import fs_agent.tools.advanced_tools  # noqa: F401
    try:
        import fs_agent.tools.search_tools  # noqa: F401
    except ImportError:
        pass

    sandbox = PathSandbox([workspace])
    set_sandbox(sandbox)

    level = PermissionLevel(permission_level)
    guard = FSPermissionGuard(level)
    registry.set_permission_guard(guard)

    model_client = OpenAICompatModel(
        api_key=api_key,
        base_url=api_base or None,
        model=model,
    )

    system_prompt = f"""你是一个专业的文件助手，能够浏览、阅读、搜索和分析用户工作区中的文件。

当用户提问时：
1. 先理解用户需要什么
2. 使用可用工具查找信息
3. 给出清晰、简洁的回答，并在相关时引用具体文件名

{INJECTION_DEFENSE_PROMPT}
"""

    budget = TokenBudget(max_context_tokens=128_000)
    loop_config = LoopConfig(max_steps=max_steps)

    return model_client, registry, system_prompt, budget, loop_config


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
st.title("🗂 Scaffold fs-agent")
st.caption(f"文档库：`{workspace_path}` · 模型：`{model_name}`")

# Render existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("meta"):
            meta = msg["meta"]
            st.caption(
                f"步数：{meta.get('steps', '—')}  |  "
                f"Tokens：{meta.get('tokens', 0):,}  "
                f"(prompt {meta.get('prompt_tokens', 0):,} + completion {meta.get('completion_tokens', 0):,})"
            )

# Chat input
user_input = st.chat_input("问我关于你文件的任何问题…")

if user_input:
    if not api_key:
        st.error("请在左侧侧边栏填写 API Key。")
        st.stop()

    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Run agent
    with st.chat_message("assistant"):
        status_box = st.status("Agent 思考中…", expanded=True)
        answer_placeholder = st.empty()

        try:
            model_client, registry, system_prompt, budget, loop_config = _build_agent_components(
                api_key=api_key,
                api_base=api_base,
                model=model_name,
                workspace=str(workspace_path),
                permission_level=permission,
                max_steps=max_steps,
            )

            from scaffold.context.window import ContextWindow
            from scaffold.loop.react import ReActLoop
            from scaffold.observability.tracer import Tracer
            from scaffold.observability.storage import TraceStorage

            # Rebuild context each turn (fresh per question, multi-turn via message history)
            context = ContextWindow(system_prompt=system_prompt, budget=budget)

            # Inject prior conversation as context
            from scaffold.models.base import Message
            for prev in st.session_state.messages[:-1]:  # exclude current user msg
                if prev["role"] == "user":
                    context.add(Message.user(prev["content"]))
                elif prev["role"] == "assistant":
                    context.add(Message.assistant(prev["content"]))

            tracer = Tracer()
            loop = ReActLoop(
                model=model_client,
                tools=registry,
                context=context,
                config=loop_config,
                tracer=tracer,
            )

            # Show live tool call updates in status box
            _original_execute = registry.execute

            async def _traced_execute(call):
                status_box.write(f"🔧 调用工具：`{call.name}` …")
                result = await _original_execute(call)
                status_box.write(f"{'❌' if result.is_error else '✅'} `{call.name}` 完成")
                return result

            registry.execute = _traced_execute

            result = asyncio.run(loop.run(user_input))

            registry.execute = _original_execute  # restore

            status_box.update(label="完成", state="complete", expanded=False)

            final_answer = result.final_message or "(Agent 未返回文本回复)"
            answer_placeholder.markdown(final_answer)

            # Token stats
            usage = result.total_usage
            st.caption(
                f"步数：{result.steps}  |  "
                f"Tokens：{usage.total_tokens:,}  "
                f"(prompt {usage.prompt_tokens:,} + completion {usage.completion_tokens:,})"
            )

            # Update session totals
            st.session_state.total_tokens += usage.total_tokens
            st.session_state.total_steps += result.steps

            # Persist message
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_answer,
                "meta": {
                    "steps": result.steps,
                    "tokens": usage.total_tokens,
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                },
            })

            # Save trace
            try:
                trace_storage = TraceStorage("traces.db")
                trace_storage.save_trace(tracer, metadata={
                    "user_input": user_input,
                    "steps": result.steps,
                })
                trace_storage.close()
            except Exception:
                pass

        except Exception as e:
            status_box.update(label="出错", state="error", expanded=True)
            error_msg = f"Agent 出错：{e}"
            answer_placeholder.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
