"""基于 Streamlit 的追踪查看器。

启动方式：
    streamlit run scaffold/observability/ui.py -- --db traces.db

展示内容：
- 带摘要统计的运行列表
- 每次运行的 span 时间线 / 瀑布图
- token 流向拆解（哪一步消耗最多）
- 错误与失败高亮
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

try:
    import streamlit as st
except ImportError:
    print("Install streamlit: pip install streamlit", file=sys.stderr)
    sys.exit(1)

from scaffold.observability.storage import TraceStorage

# ---------------------------------------------------------------------------
# 页面配置
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Scaffold Trace Viewer", page_icon="🔍", layout="wide")


@st.cache_resource
def get_storage(db_path: str) -> TraceStorage:
    return TraceStorage(db_path)


def _fmt_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _fmt_latency(ms: float) -> str:
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms / 1000:.2f}s"


# ---------------------------------------------------------------------------
# 侧边栏 —— 数据库选择与运行列表
# ---------------------------------------------------------------------------
st.sidebar.title("🔍 Scaffold Trace Viewer")

db_path = st.sidebar.text_input("Trace DB path", value="traces.db")
if not Path(db_path).exists():
    st.warning(f"Database `{db_path}` not found. Run fs-agent first to generate traces.")
    st.stop()

storage = get_storage(db_path)
runs = storage.list_runs(limit=100)

if not runs:
    st.info("No traces recorded yet. Use fs-agent to generate some runs.")
    st.stop()

st.sidebar.markdown(f"**{len(runs)} runs**")

run_options = {
    f"{_fmt_ts(r['created_at'])} — {r['run_id']}": r["run_id"]
    for r in runs
}
selected_label = st.sidebar.selectbox("Select a run", list(run_options.keys()))
selected_run_id = run_options[selected_label]

# ---------------------------------------------------------------------------
# 主区域：运行概览
# ---------------------------------------------------------------------------
run_meta = next(r for r in runs if r["run_id"] == selected_run_id)
spans = storage.get_spans(selected_run_id)

st.title(f"Run: `{selected_run_id}`")

col1, col2, col3, col4 = st.columns(4)
meta = run_meta.get("metadata", {})
if isinstance(meta, str):
    meta = json.loads(meta)

col1.metric("Steps", meta.get("steps", "—"))
col2.metric("Spans", len(spans))

total_latency = sum(s.get("latency_ms", 0) for s in spans)
col3.metric("Total Latency", _fmt_latency(total_latency))

total_tokens = sum(
    s.get("metadata", {}).get("prompt_tokens", 0) + s.get("metadata", {}).get("completion_tokens", 0)
    for s in spans
)
col4.metric("Total Tokens", f"{total_tokens:,}")

if "user_input" in meta:
    st.markdown(f"**User Input:** {meta['user_input']}")

# ---------------------------------------------------------------------------
# Span 时间线（瀑布图）
# ---------------------------------------------------------------------------
st.subheader("Span Timeline")

if spans:
    min_start = min(s["start_time"] for s in spans if s["start_time"])
    for s in spans:
        smeta = s.get("metadata", {})
        if isinstance(smeta, str):
            smeta = json.loads(smeta)

        offset = (s["start_time"] - min_start) * 1000 if s["start_time"] else 0
        latency = s.get("latency_ms", 0)

        # 按类型设置颜色
        kind = s["kind"]
        color_map = {"agent": "🟢", "llm": "🔵", "tool": "🟠", "other": "⚪"}
        icon = color_map.get(kind, "⚪")

        # 子 span 缩进显示
        indent = "  └─ " if s.get("parent_id") else ""

        status_icon = "✅" if s["status"] == "completed" else "❌"
        tokens_info = ""
        if "prompt_tokens" in smeta:
            tokens_info = f" | {smeta['prompt_tokens']}+{smeta.get('completion_tokens', 0)} tok"

        st.text(
            f"{icon} {indent}{s['name']:30s} {status_icon} "
            f"{_fmt_latency(latency):>8s}{tokens_info}"
        )

# ---------------------------------------------------------------------------
# Token 流向
# ---------------------------------------------------------------------------
st.subheader("Token Flow by Step")

llm_spans = [s for s in spans if s["kind"] == "llm"]
if llm_spans:
    chart_data = []
    for s in llm_spans:
        smeta = s.get("metadata", {})
        if isinstance(smeta, str):
            smeta = json.loads(smeta)
        chart_data.append({
            "step": s["name"],
            "prompt_tokens": smeta.get("prompt_tokens", 0),
            "completion_tokens": smeta.get("completion_tokens", 0),
        })

    st.bar_chart(
        data={
            "Prompt Tokens": [d["prompt_tokens"] for d in chart_data],
            "Completion Tokens": [d["completion_tokens"] for d in chart_data],
        },
    )

    # 表格视图
    st.dataframe(chart_data, use_container_width=True)
else:
    st.info("No LLM spans in this run.")

# ---------------------------------------------------------------------------
# 工具调用详情
# ---------------------------------------------------------------------------
st.subheader("Tool Calls")

tool_spans = [s for s in spans if s["kind"] == "tool"]
if tool_spans:
    for s in tool_spans:
        smeta = s.get("metadata", {})
        if isinstance(smeta, str):
            smeta = json.loads(smeta)

        is_error = smeta.get("is_error", False)
        status = "❌ Error" if is_error else "✅ OK"
        result_len = smeta.get("result_length", 0)

        with st.expander(f"{s['name']} — {status} ({_fmt_latency(s.get('latency_ms', 0))})"):
            st.json(smeta)
else:
    st.info("No tool calls in this run.")

# ---------------------------------------------------------------------------
# 原始 span 数据
# ---------------------------------------------------------------------------
with st.expander("Raw Span Data"):
    st.json(spans)
