# Scaffold — 通用 Agent Harness + 文件助手

> **项目定位**：这不是一个普通的"能问答文件的聊天机器人"，而是一套**通用 Agent Harness 框架（Scaffold）**，以文件系统助手（fs-agent）作为首个落地应用，验证 Harness 在工具设计、上下文管理、可观测性、安全沙箱上的工程实践。

---

## 目录

- [整体架构](#整体架构)
- [快速开始](#快速开始)
- [运行方式](#运行方式)
  - [Web UI 聊天界面（推荐）](#web-ui-聊天界面推荐)
  - [命令行 CLI](#命令行-cli)
  - [Trace 可视化界面](#trace-可视化界面)
  - [评估框架](#评估框架)
- [fs-agent 工具集（15 个工具）](#fs-agent-工具集15-个工具)
- [Harness 核心模块](#harness-核心模块)
- [安全机制](#安全机制)
- [配置说明](#配置说明)
- [项目结构](#项目结构)

---

## 整体架构

```
┌─────────────────────────────────────────────────────┐
│  应用层：fs-agent（文件系统 Agent）                   │
│  · 15 个文件工具  · 三级权限控制  · CLI & Web UI     │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│  Harness 层：Scaffold（核心框架）                     │
│  · Agent Loop     · Tool Runtime   · Context Manager │
│  · Observability  · Safety Layer   · Cache           │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│  模型层：LLM Adapter                                  │
│  · 统一接口，支持 GLM / MiniMax / DeepSeek 等         │
└─────────────────────────────────────────────────────┘
```

三层严格分离：`scaffold/` 不依赖 `fs_agent/`，Harness 可复用于任何 Agent 应用。

---

## 快速开始

### 1. 克隆并安装依赖

```bash
git clone https://github.com/abc688-shawn/scaffold-agent-harness.git
cd scaffold-agent-harness

# 安装 uv（如果还没有）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 一条命令完成：创建虚拟环境 + 安装所有依赖（含 PDF/DOCX 解析和 Web UI）
uv sync --all-extras
```

### 2. 配置 API Key

复制配置模板并填写你的 LLM API 信息：

```bash
cp .env.example .env
```

编辑 `.env`：

```bash
# 智谱 GLM（推荐）
OPENAI_API_BASE=https://open.bigmodel.cn/api/paas/v4/
OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL_NAME=glm-5

# 或 MiniMax M2
# OPENAI_API_BASE=https://api.minimax.chat/v1
# OPENAI_API_KEY=your-key
# OPENAI_MODEL_NAME=MiniMax-M2

# 或 DeepSeek
# OPENAI_API_BASE=https://api.deepseek.com/v1
# OPENAI_API_KEY=your-key
# OPENAI_MODEL_NAME=deepseek-chat
```

### 3. 验证安装

```bash
uv run pytest                                                    # 运行 121 个单元测试
uv run python -m evals.runner --cases evals/cases/ --dry-run    # 预览 52 个评估用例
```

---

## 运行方式

### Web UI 聊天界面（推荐）

```bash
uv run streamlit run fs_agent/app.py
```

浏览器打开 `http://localhost:8501`

**界面功能：**

| 区域 | 说明 |
|------|------|
| 左侧边栏 · API 配置 | 填写 API Key、Base URL、模型名，支持运行时切换 |
| 左侧边栏 · 文档库 | 拖拽上传文件，**永久保存**到本地 `workspace/` 目录 |
| 左侧边栏 · 会话统计 | 显示本次对话累计 Token 用量和 Agent 步数 |
| 主区域 · 对话框 | 多轮对话，实时显示 Agent 调用了哪些工具 |
| 每条回复下方 | 显示该轮的步数、prompt tokens、completion tokens |

**使用流程：**

1. 在侧边栏上传 PDF、DOCX、TXT 等文件
2. 在底部输入框提问，例如：
   - `"帮我总结一下这个 PDF 的主要内容"`
   - `"对比这两篇文档，找出观点差异"`
   - `"在所有文件中搜索关于'机器学习'的内容"`
   - `"把 downloads 目录里的文件按类型整理一下"`

> 上传的文件保存在 `workspace/` 目录，重启应用后仍然保留。也可以在侧边栏手动修改库目录路径，指向本机任意文件夹。

---

### 命令行 CLI

```bash
python -m fs_agent.cli \
  --workspace ~/Documents \
  --model glm-5 \
  --permission confirm_write
```

**参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--workspace` | `.` | Agent 操作的根目录（沙箱范围） |
| `--api-key` | 读取 `OPENAI_API_KEY` 环境变量 | LLM API Key |
| `--api-base` | 读取 `OPENAI_API_BASE` 环境变量 | API Base URL |
| `--model` | 读取 `OPENAI_MODEL_NAME` 环境变量 | 模型名称 |
| `--permission` | `confirm_write` | 权限级别（见下方说明） |
| `--max-steps` | `15` | 单次问答最大 Agent 步数 |
| `--trace-db` | `traces.db` | Trace 数据库路径 |
| `--embed-model` | 无 | 指定后启用语义搜索功能 |

---

### Trace 可视化界面

每次 Agent 运行都会自动记录 Trace 数据到 `traces.db`，可用可视化界面查看：

```bash
uv run streamlit run scaffold/observability/ui.py -- --db traces.db
```

浏览器打开 `http://localhost:8501`，可查看：

- 每次运行的 **Span 时间线**（瀑布图）
- 各步骤的 **Token 消耗分布**（柱状图）
- 每个工具调用的详细元数据（耗时、是否出错、返回长度）
- 原始 Span JSON 数据

---

### 评估框架

```bash
# 预览所有评估用例（不实际运行）
uv run python -m evals.runner --cases evals/cases/ --dry-run

# 规则评估
uv run python -m evals.runner \
  --cases evals/cases/ \
  --api-key $OPENAI_API_KEY \
  --base-url https://open.bigmodel.cn/api/paas/v4/ \
  --model glm-5 \
  --output results.json

# 启用 LLM-as-judge（更精准的评分）
uv run python -m evals.runner --cases evals/cases/ --judge --output results.json

# 只跑某个分类
uv run python -m evals.runner --cases evals/cases/ --category security
```

---

## fs-agent 工具集（15 个工具）

### 基础文件工具

| 工具名 | 说明 | 主要参数 |
|--------|------|----------|
| `list_files` | 列出目录中的文件，显示大小、修改时间 | `path`, `recursive`, `pattern`（支持 glob） |
| `read_file` | 读取文本文件内容，支持分块读取大文件 | `path`, `offset`, `length` |
| `search_files` | 在目录中全文搜索，返回匹配行 | `query`, `path`, `pattern`, `max_results` |
| `file_info` | 获取文件详细元数据（大小、类型、创建/修改时间） | `path` |
| `write_file` | 写入或追加文件内容 | `path`, `content`, `mode`（overwrite/append） |
| `move_file` | 移动或重命名文件 | `source`, `destination` |
| `delete_file` | 删除文件（需权限确认） | `path` |

### 文档解析工具

| 工具名 | 说明 | 主要参数 |
|--------|------|----------|
| `read_pdf` | 读取 PDF 文件文本，支持按页范围提取 | `path`, `pages`（如 `"1-5"` 或 `"all"`）, `max_pages` |
| `read_docx` | 读取 Word 文档，保留标题结构，提取表格 | `path`, `max_paragraphs` |
| `preview_file` | 智能预览任意格式文件（自动识别 PDF/DOCX/CSV/JSON/代码等） | `path` |
| `summarize_file` | 提取文件结构信息（PDF 目录、代码函数名、CSV 列名等），不读全文 | `path` |

### 高级分析工具

| 工具名 | 说明 | 主要参数 |
|--------|------|----------|
| `organize_files` | 按扩展名/日期/大小规划或执行文件整理，默认 dry-run 先展示计划 | `source`, `strategy`, `dry_run` |
| `compare_files` | 对比两个文件的内容差异 | `path_a`, `path_b` |
| `tag_files` | 给文件打标签，写入元数据 JSON | `path`, `tags` |
| `search_by_tag` | 按标签检索文件 | `tags`, `path` |

### 语义搜索工具（需配置 embedding 模型）

| 工具名 | 说明 |
|--------|------|
| `index_files` | 对工作区文件建立向量索引 |
| `semantic_search` | 用自然语言语义搜索文件内容（余弦相似度） |

---

## Harness 核心模块

### LLM Adapter（`scaffold/models/`）

统一 `ChatModel` 接口，屏蔽各厂商 API 差异：
- 支持任何 **OpenAI 兼容协议**（GLM、MiniMax、DeepSeek、Moonshot 等）
- 自动处理 **streaming tool call 分块拼接**
- 内置 **指数退避重试**（超时、限流、5xx 自动重试，最多 4 次）
- `MockModel`：按脚本返回预设响应，用于 eval 离线回放

### Tool Runtime（`scaffold/tools/`）

- `@tool` 装饰器：从函数签名和 docstring **自动生成** OpenAI function schema
- 工具执行：同步函数自动转异步（`asyncio.to_thread`），支持并发执行多个工具
- 标准化错误码：`NOT_FOUND` / `PATH_OUTSIDE_SANDBOX` / `FILE_TOO_LARGE` / `UNSUPPORTED_FORMAT` / `INTERNAL_ERROR`，每种错误附带给模型的修复提示

### Context Manager（`scaffold/context/`）

- **Token 预算分配**：总预算按 system + tools + history + response_reserve 切分
- **两种压缩策略**：
  - `SlidingWindow`：保留最近 N 条消息，截断旧消息
  - `SummaryWithRefs`：旧工具结果转引用 ID 存储，摘要代替原文，按需回取
- **KV Cache 友好布局**：稳定的 system prompt 前置，动态 history 后置，最大化 prefix cache 命中
- **动态 System Prompt**：支持 planning → execution 阶段切换不同指令

### Agent Loop（`scaffold/loop/`）

ReAct（Reasoning + Acting）标准实现：
- **双重预算**：最大步数 + 最大 Token 用量，任一触发则终止
- **循环检测**：连续 3 步 tool call 参数完全相同，自动注入 reflection 提示让模型换策略
- 每步 LLM call 和工具调用都写入 Trace

### Observability（`scaffold/observability/`）

- `Tracer`：每次 agent run 构建一棵 Span 树，记录 latency / token / status
- `TraceStorage`：SQLite 持久化，保存所有历史运行
- `ui.py`：Streamlit 可视化界面，展示时间线、Token 流向、工具详情

### Cache（`scaffold/cache/`）

- 基于 TTL 的 in-memory 缓存
- 以工具名 + 参数 hash 为 key，相同调用不重复执行
- 统计 hit / miss 率，可用于 eval 零成本回放

---

## 安全机制

### 路径沙箱（Path Sandbox）

所有文件操作都受 `PathSandbox` 限制，Agent **只能访问指定工作区目录**，路径穿越（如 `../../etc/passwd`）会直接返回错误，不会执行。

### Prompt Injection 防御

两道防线：
1. 所有工具返回内容用 `<tool_result>` 标签包裹，并转义内部嵌套标签
2. System prompt 明确告知模型：`<tool_result>` 里的内容是**数据**，不是指令，发现可疑注入立即上报

### 敏感信息脱敏

自动检测并 mask 以下内容：
- API Key / Token（正则匹配 `sk-...` 等格式）
- 中国身份证号（18位）
- 手机号（1开头11位）
- 邮箱地址
- 密码字段（`password=...` 等）

### 三级权限控制

| 级别 | 说明 |
|------|------|
| `read_only` | 只允许读取类工具，任何写操作直接拒绝 |
| `confirm_write`（默认） | 读操作自由执行，写/移动/删除操作需用户在终端确认 |
| `autonomous` | 所有操作自主执行，无需确认（适合受信任的自动化场景） |

---

## 配置说明

### 环境变量（`.env`）

| 变量 | 说明 | 示例 |
|------|------|------|
| `OPENAI_API_KEY` | LLM API Key | `your-key-here` |
| `OPENAI_API_BASE` | API Base URL | `https://open.bigmodel.cn/api/paas/v4/` |
| `OPENAI_MODEL_NAME` | 模型名称 | `glm-5` |

### 可选依赖

```bash
uv sync --all-extras              # 安装全部可选依赖（推荐）
uv sync --extra fs --extra ui     # 只安装指定依赖组
```

---

## 项目结构

```
scaffold-agent-harness/
│
├── scaffold/                     # 核心 Harness 框架（可复用）
│   ├── models/
│   │   ├── base.py               # 统一 ChatModel 接口、Message、ToolCall 等
│   │   ├── openai_compat.py      # OpenAI 兼容适配器（含 streaming + 重试）
│   │   └── mock.py               # 测试用 MockModel，支持脚本化回放
│   ├── tools/
│   │   ├── registry.py           # ToolRegistry、@tool 装饰器、并发执行
│   │   ├── schema.py             # 函数签名 → OpenAI schema 自动生成
│   │   └── errors.py             # 标准化错误码
│   ├── context/
│   │   ├── budget.py             # Token 预算分配
│   │   ├── compression.py        # 消息压缩（滑动窗口 + 引用 ID）
│   │   └── window.py             # ContextWindow，KV cache 友好布局
│   ├── loop/
│   │   └── react.py              # ReAct 循环，双重预算，循环检测
│   ├── safety/
│   │   ├── sandbox.py            # 路径沙箱
│   │   ├── injection.py          # Prompt Injection 防御
│   │   └── redaction.py          # 敏感信息正则检测与脱敏
│   ├── observability/
│   │   ├── tracer.py             # Span 树 Tracer
│   │   ├── storage.py            # SQLite 持久化
│   │   └── ui.py                 # Streamlit Trace 可视化界面
│   └── cache/
│       └── cache.py              # TTL in-memory 缓存
│
├── fs_agent/                     # 文件系统 Agent（参考实现）
│   ├── app.py                    # Streamlit 聊天 Web UI 入口
│   ├── cli.py                    # 命令行交互入口
│   ├── tools/
│   │   ├── file_tools.py         # 基础文件工具（7 个）
│   │   ├── doc_tools.py          # 文档解析工具（4 个）
│   │   ├── advanced_tools.py     # 高级工具（4 个）
│   │   └── search_tools.py       # 语义搜索工具（需 embedding 模型）
│   └── policies/
│       └── permissions.py        # 三级权限守卫
│
├── evals/                        # 评估框架
│   ├── runner.py                 # 评估执行器（规则 + LLM-as-judge）
│   ├── cases/                    # 52+ YAML 评估用例
│   │   ├── fs_basic.yaml         # 基础文件操作
│   │   ├── fs_security.yaml      # 安全对抗（injection、路径穿越）
│   │   ├── fs_tool_select.yaml   # 工具选择准确性
│   │   ├── fs_context.yaml       # 跨文件分析
│   │   ├── fs_doc_org.yaml       # 文档整理
│   │   └── fs_advanced.yaml      # 高级功能
│   └── judges/
│       ├── llm_judge.py          # LLM-as-judge 评分器
│       └── prompts.py            # 评分提示词
│
├── tests/                        # 单元测试（121 个）
├── workspace/                    # 文档库（Web UI 上传的文件存放于此，不纳入 git）
├── .env.example                  # 环境变量配置模板
└── pyproject.toml                # 项目依赖与入口点
```

---

## 技术栈

| 组件 | 选型 | 说明 |
|------|------|------|
| LLM 接入 | OpenAI SDK | 兼容 GLM / MiniMax / DeepSeek 等所有 OpenAI 协议厂商 |
| 重试 | tenacity | 指数退避，处理限流和超时 |
| Token 计数 | tiktoken | cl100k_base 编码，支持中文估算 fallback |
| PDF 解析 | PyMuPDF (fitz) | 速度快，支持 TOC 提取 |
| DOCX 解析 | python-docx | 保留标题层级，提取表格 |
| 编码检测 | chardet | 自动识别文件编码 |
| Trace 存储 | SQLite | 零依赖，本地持久化 |
| Web UI | Streamlit | 快速构建交互界面 |
| 测试 | pytest + pytest-asyncio | 异步测试支持 |
| 代码检查 | ruff | 快速 linter |
