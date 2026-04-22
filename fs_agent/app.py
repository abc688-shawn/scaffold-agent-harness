"""Scaffold fs-agent —— 交互式网页聊天界面。

启动方式：
    streamlit run fs_agent/app.py

功能：
- 将文档（PDF、DOCX、TXT、CSV 等）上传到本地持久化文档库
- 围绕你的文件与 fs-agent 对话
- 侧边栏展示 API 配置、文档库与每轮 token 统计
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# 加载 .env，避免每次都手动填写 API 凭据
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env", override=False)
except ImportError:
    pass

try:
    import streamlit as st
except ImportError:
    print("Install streamlit: pip install streamlit", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# 页面配置 —— 必须是第一次 Streamlit 调用
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Scaffold fs-agent",
    page_icon="🗂",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 默认的持久化文档库目录 —— 位于项目根目录
_DEFAULT_WORKSPACE = Path(__file__).parent.parent / "workspace"

# ---------------------------------------------------------------------------
# Session state 辅助函数
# ---------------------------------------------------------------------------

def _init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []   # 形如 {"role", "content", "meta"} 的消息列表
    if "total_tokens" not in st.session_state:
        st.session_state.total_tokens = 0
    if "total_steps" not in st.session_state:
        st.session_state.total_steps = 0

_init_state()

# ---------------------------------------------------------------------------
# 侧边栏 —— 配置与文件上传
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🗂 fs-agent")
    st.caption("Scaffold · Agent Harness Demo")
    st.divider()

    # ── API 配置 ────────────────────────────────────────────────────────
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
        ["autonomous", "read_only"],
        index=0,
        help="autonomous: Web UI 中推荐，所有操作直接执行 | read_only: 只允许读取，禁止任何写操作（confirm_write 需要终端交互，不适用于 Web UI）",
    )
    max_steps = st.slider("最大步数", min_value=5, max_value=30, value=15)

    st.divider()

    # ── 文档库 ──────────────────────────────────────────────────────────
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

    # 将上传文件保存到持久化工作区
    if uploaded:
        for uf in uploaded:
            dest = workspace_path / uf.name
            dest.write_bytes(uf.read())   # 覆盖写入，等价于更新已有文件
            st.toast(f"已保存: {uf.name}", icon="✅")

    # 列出文档库中的所有文件
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

    # ── 统计信息 ────────────────────────────────────────────────────────
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
# Agent 构建器（按配置缓存）
# ---------------------------------------------------------------------------

@st.cache_resource(hash_funcs={"builtins.str": lambda s: s})
def _build_agent_components(api_key: str, api_base: str, model: str,
                             workspace: str, permission_level: str, max_steps: int):
    """构建并返回 agent 组件；在配置变化前都会复用缓存。"""
    from scaffold.cache.cache import ResultCache
    from scaffold.context.budget import TokenBudget
    from scaffold.loop.react import LoopConfig
    from scaffold.models.openai_compat import OpenAICompatModel
    from scaffold.prompts.loader import build_dynamic_prompt
    from scaffold.safety.injection import INJECTION_DEFENSE_PROMPT
    from scaffold.safety.sandbox import PathSandbox
    from fs_agent.tools.file_tools import registry, set_sandbox
    from fs_agent.policies.permissions import PermissionLevel, FSPermissionGuard
    import fs_agent.tools.doc_tools       # noqa: F401 — 导入即完成工具注册
    import fs_agent.tools.advanced_tools  # noqa: F401 — 导入即完成工具注册
    try:
        import fs_agent.tools.search_tools  # noqa: F401 — 若依赖存在则注册搜索工具
    except ImportError:
        pass

    sandbox = PathSandbox([workspace])
    set_sandbox(sandbox)

    level = PermissionLevel(permission_level)
    guard = FSPermissionGuard(level)
    registry.set_permission_guard(guard)

    result_cache = ResultCache(default_ttl=300.0)
    registry.set_cache(result_cache)

    model_client = OpenAICompatModel(
        api_key=api_key,
        base_url=api_base or None,
        model=model,
    )

    dynamic_prompt = build_dynamic_prompt(
        "system/fs_agent.j2",
        workspace=workspace,
        injection_defense=INJECTION_DEFENSE_PROMPT,
    )

    budget = TokenBudget(max_context_tokens=128_000)
    loop_config = LoopConfig(max_steps=max_steps)

    return model_client, registry, dynamic_prompt, budget, loop_config, result_cache


# ---------------------------------------------------------------------------
# 主聊天区域
# ---------------------------------------------------------------------------
st.title("🗂 Scaffold fs-agent")
st.caption(f"文档库：`{workspace_path}` · 模型：`{model_name}`")

# 渲染已有消息
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

# 聊天输入框
user_input = st.chat_input("问我关于你文件的任何问题…")

if user_input:
    if not api_key:
        st.error("请在左侧侧边栏填写 API Key。")
        st.stop()

    # 立即展示用户消息
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 运行 agent
    with st.chat_message("assistant"):
        status_box = st.status("Agent 思考中…", expanded=True)
        answer_placeholder = st.empty()

        try:
            model_client, registry, dynamic_prompt, budget, loop_config, result_cache = _build_agent_components(
                api_key=api_key,
                api_base=api_base,
                model=model_name,
                workspace=str(workspace_path),
                permission_level=permission,
                max_steps=max_steps,
            )

            from scaffold.context.window import ContextWindow
            from scaffold.loop.middlewares import CostTrackerMiddleware, RedactionMiddleware, ToolCallLimitMiddleware
            from scaffold.loop.react import ReActLoop
            from scaffold.observability.storage import TraceStorage
            from scaffold.observability.tracer import Tracer

            # 每轮重建上下文（每个问题使用新的上下文，多轮效果通过历史消息注入实现）
            context = ContextWindow(system_prompt=dynamic_prompt, budget=budget)

            middlewares = [
                ToolCallLimitMiddleware(repeat_limit=3),
                RedactionMiddleware(),
                CostTrackerMiddleware(warn_fraction=0.8, result_cache=result_cache),
            ]

            # 将既有对话注入为上下文
            from scaffold.models.base import Message
            for prev in st.session_state.messages[:-1]:  # 排除当前这条用户消息
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
                middlewares=middlewares,
            )

            # 在状态框中实时显示工具调用进度
            _original_execute = registry.execute

            async def _traced_execute(call):
                status_box.write(f"🔧 调用工具：`{call.name}` …")
                result = await _original_execute(call)
                status_box.write(f"{'❌' if result.is_error else '✅'} `{call.name}` 完成")
                return result

            registry.execute = _traced_execute

            result = asyncio.run(loop.run(user_input))

            registry.execute = _original_execute  # 恢复原始实现

            status_box.update(label="完成", state="complete", expanded=False)

            final_answer = result.final_message or "(Agent 未返回文本回复)"
            answer_placeholder.markdown(final_answer)

            # Token 统计
            usage = result.total_usage
            st.caption(
                f"步数：{result.steps}  |  "
                f"Tokens：{usage.total_tokens:,}  "
                f"(prompt {usage.prompt_tokens:,} + completion {usage.completion_tokens:,})"
            )

            # 更新会话累计统计
            st.session_state.total_tokens += usage.total_tokens
            st.session_state.total_steps += result.steps

            # 持久化消息
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

            # 保存追踪信息
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
