# Scaffold — 通用 Agent Harness + 文件助手

> **项目定位**：这不是一个普通的"能问答文件的聊天机器人"，而是一套**通用 Agent Harness 框架（Scaffold）**，以文件系统助手（fs-agent）作为首个落地应用，验证 Harness 在工具设计、上下文管理、Middleware 管道、技能系统、断点续跑、可观测性、安全沙箱上的完整工程实践。

---

## 目录

- [整体架构](#整体架构)
- [快速开始](#快速开始)
- [运行方式](#运行方式)
- [工具集（21 个工具）](#工具集21-个工具)
- [Skills 技能系统](#skills-技能系统)
- [Harness 核心模块](#harness-核心模块)
- [安全机制](#安全机制)
- [配置说明](#配置说明)
- [项目结构](#项目结构)
- [技术栈](#技术栈)

---

## 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│  应用层：fs-agent（文件系统 Agent）                                │
│  · 21 个工具  · 三级权限  · Skills 技能系统  · CLI & Web UI         │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  Harness 层：Scaffold（核心框架）                                  │
│  · ReAct Loop      · Middleware 管道   · Context Manager         │
│  · Prompt 模板     · Skills Loader     · Checkpoint / Resume     │
│  · Tool Runtime    · Observability     · Safety Layer           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  模型层：LLM Adapter                                             │
│  · 统一 ChatModel 接口，支持 GLM / MiniMax / DeepSeek 等           │
└─────────────────────────────────────────────────────────────────┘
```

三层严格单向依赖：`fs_agent/` → `scaffold/` → 外部库。`scaffold/` 永远不导入 `fs_agent/`，Harness 可独立复用于其他 Agent 应用。

---

## 快速开始

### 1. 克隆并安装依赖

```bash
git clone https://github.com/abc688-shawn/scaffold-agent-harness.git
cd scaffold-agent-harness

# 安装 uv（如果还没有）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 一条命令完成：创建虚拟环境 + 安装所有依赖（含文档解析和 Web UI）
uv sync --all-extras
```

> 需要 Python 3.10+。uv 会自动管理 Python 版本。

### 2. 配置 API Key

编辑 `.env`（取消注释你使用的服务）：

```bash
# LLM — 选一个取消注释
OPENAI_API_BASE=https://open.bigmodel.cn/api/paas/v4/
OPENAI_API_KEY=your-llm-key
OPENAI_MODEL_NAME=glm-5

# Embedding（可选，启用后支持文档语义检索）
EMBEDDING_API_BASE=https://api.siliconflow.cn/v1
EMBEDDING_API_KEY=your-embedding-key
EMBEDDING_MODEL_NAME=Qwen/Qwen3-Embedding-8B
```

### 3. 验证安装

```bash
uv run pytest                                                    # 159 个单元测试
uv run python -m evals.runner --cases evals/cases/ --dry-run    # 预览 52 个评估用例
```

---

## 运行方式

### Web UI 聊天界面（推荐）

```bash
uv run streamlit run fs_agent/app.py
```

浏览器打开 `http://localhost:8501`

| 区域 | 说明 |
|------|------|
| 侧边栏 · API 配置 | 运行时填写 / 切换 API Key、Base URL、模型 |
| 侧边栏 · 文档库 | 拖拽上传文件，持久保存到本地 `workspace/` |
| 侧边栏 · 未完成的运行 | 显示中断的历史任务，一键续跑 |
| 侧边栏 · 会话统计 | 本次对话累计 Token 和步数 |
| 主区域 · 对话框 | 多轮对话，实时显示工具调用过程 |

### 命令行 CLI

```bash
uv run fs-agent \
  --workspace ~/Documents \
  --model glm-5 \
  --permission confirm_write
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--workspace` | `.` | Agent 操作根目录（沙箱范围） |
| `--api-key` | `OPENAI_API_KEY` 环境变量 | LLM API Key |
| `--api-base` | `OPENAI_API_BASE` 环境变量 | API Base URL |
| `--model` | `OPENAI_MODEL_NAME` 环境变量 | 模型名称 |
| `--permission` | `confirm_write` | 权限级别 |
| `--max-steps` | `15` | 单次问答最大步数 |
| `--embed-model` | — | 启用语义搜索（需同时指定 `--embed-key`） |

### Trace 可视化界面

每次运行自动记录 Trace 到 `traces.db`：

```bash
uv run streamlit run scaffold/observability/ui.py -- --db traces.db
```

可查看 Span 时间线（瀑布图）、Token 消耗分布、工具调用详情。

### 评估框架

```bash
# 只做规则评估（免费，无需 LLM 调用）
uv run python -m evals.runner --cases evals/cases/ --output results.json

# 加 LLM-as-judge 评分
uv run python -m evals.runner --cases evals/cases/ --judge --output results.json

# 只跑某个分类
uv run python -m evals.runner --cases evals/cases/ --category security
```

---

## 工具集（21 个工具）

### 基础文件工具（`file_tools.py`）

| 工具名 | 说明 |
|--------|------|
| `list_files` | 列出目录文件，支持 glob 模式和递归 |
| `read_file` | 读取文本文件，支持 offset / length 分块 |
| `write_file` | 写入或追加文件内容 |
| `search_files` | 全文搜索，返回匹配行及上下文 |
| `file_info` | 获取文件详细元数据（大小、类型、时间） |
| `move_file` | 移动或重命名文件 |
| `rename_file` | 原地重命名文件 |
| `delete_file` | 删除文件（需权限确认） |
| `make_directory` | 创建目录（支持嵌套） |

### 文档处理工具（`doc_tools.py`）

使用 [MarkItDown](https://github.com/microsoft/markitdown) 统一将 PDF / DOCX / PPTX / XLSX / HTML 等格式转换为 Markdown。

| 工具名 | 说明 |
|--------|------|
| `read_document` | 读取文档全文并转为 Markdown，适合总结类任务 |
| `search_document` | 按语义检索文档中最相关的段落——无需读全文。首次调用自动建向量索引，后续复用缓存（需配置 Embedding 服务） |
| `preview_file` | 智能预览任意格式文件（文档 / CSV / JSON / 代码） |
| `summarize_file` | 提取文件结构信息（文档标题、代码函数名、CSV 列名），不读全文 |

`read_document` vs `search_document` 选择原则：

- 总结整篇文档、文档较小 → `read_document`
- 用户问的是文档中具体内容、文档超过 20KB → `search_document`

### 高级分析工具（`advanced_tools.py`）

| 工具名 | 说明 |
|--------|------|
| `organize_files` | 按扩展名 / 日期 / 大小规划或执行文件整理，默认先 dry-run 展示计划 |
| `compare_files` | 对比两个文件内容差异 |
| `tag_files` | 给文件打标签，写入同目录 `.file_tags.json` |
| `search_by_tag` | 按标签检索文件 |

### 语义搜索工具（`search_tools.py`）

| 工具名 | 说明 |
|--------|------|
| `index_files` | 对工作区文件建立向量索引（需配置 Embedding 服务） |
| `semantic_search` | 用自然语言语义搜索已索引文件 |

### 系统工具

| 工具名 | 说明 |
|--------|------|
| `retrieve_reference` | 按 ID 回取被压缩摘要的原始工具结果（`SUMMARY_WITH_REFS` 压缩策略配套工具） |
| `list_skills` | 列出当前 Agent 加载的所有技能及其触发关键词 |

---

## Skills 技能系统

Skills 是从 Markdown 文件中加载的"领域操作手册"。当用户输入命中触发关键词时，`SkillTriggerMiddleware` 自动将该技能的正文注入 system prompt，引导模型按规定框架处理任务。

**核心优势**：改进 Agent 行为只需编辑 SKILL.md，无需修改代码；可用 Git 追踪每次行为变化。

### 内置技能（`fs_agent/skills/`）

| 技能 | 触发关键词（示例） | 主要工具 |
|------|-------------------|----------|
| `file-organize` | 整理、归类、归档、分组 | `organize_files`, `move_file`, `make_directory` |
| `doc-qa` | 总结、摘要、讲什么、归纳、解读 | `read_document`, `search_document`, `summarize_file` |
| `batch-compare` | 对比、比较、找差异、区别 | `compare_files`, `read_document` |

### 自定义技能

在 `fs_agent/skills/<skill-name>/SKILL.md` 新建文件，格式：

```markdown
---
name: my-skill
description: 当用户... 时触发
trigger-keywords: [关键词1, 关键词2]
allowed-tools: tool_a tool_b
metadata:
  version: "1.0"
---

# 操作手册正文（何时使用 / 核心框架 / 输出格式建议）
...
```

Agent 重启后自动加载，无需任何代码改动。

---

## Harness 核心模块

### ReAct Loop（`scaffold/loop/react.py`）

标准 ReAct（Reasoning + Acting）实现：
- 双重终止预算：最大步数 + 最大 Token 用量，任一触发则终止并返回当前结果
- 工具并发执行：同一步骤的多个工具调用通过 `asyncio.gather` 并发运行
- Middleware 管道：每步经过 `before_step → after_llm → after_tool → after_step` 四个切点

### Middleware 管道（`scaffold/loop/middleware.py`）

横切关注点全部通过 Middleware 实现，主循环零污染：

| Middleware | 切点 | 功能 |
|---|---|---|
| `ToolCallLimitMiddleware` | `after_tool` | 按 `(tool_name, args_hash)` 计数重复调用；达到阈值注入 reflection 提示，防止死循环 |
| `RedactionMiddleware` | `after_tool` | 正则扫描工具返回内容，自动 mask API Key、身份证号、手机号、邮箱等敏感信息 |
| `CostTrackerMiddleware` | `after_step` | 累计 Token 用量，超过预算分数时触发告警 |
| `SkillTriggerMiddleware` | `before_step` | step=1 时扫描用户消息，命中关键词则注入技能正文到 EXECUTION 阶段 system prompt |

每次 `FSAgent.run()` 创建全新 Middleware 栈，确保计数器从零开始，多轮对话不串台。

### 断点续跑（`scaffold/loop/checkpoint.py`）

任务中断（浏览器刷新、进程崩溃）后可从断点恢复：

```python
agent = FSAgent(config)

# 查看未完成的运行
runs = agent.list_incomplete_runs()

# 从指定 run_id 恢复
result = await agent.resume(run_id="abc123")
```

实现细节：
- 每个工具步骤完成后执行 SQLite UPSERT（~1ms），代价极低
- 保存点选在"工具结果写入 context 之后"，恢复时无需重复执行任何工具（包括非幂等操作）
- 恢复时跳过重复添加用户消息，回填已用 Token 计数以保证预算判断准确

Web UI 侧边栏的"未完成的运行"列表可直接一键续跑。

### Context Manager（`scaffold/context/`）

- **Token 预算分配**：总预算切分为 system + tools schema + history + response_reserve
- **两种压缩策略**：
  - `SlidingWindow`：保留最近 N 条消息，截断旧消息
  - `SummaryWithRefs`：旧工具结果按引用 ID 存入 `ReferenceStore`，摘要代替原文；Agent 后续可通过 `retrieve_reference` 工具按需回取完整内容
- **动态 System Prompt**：Jinja2 模板 + 阶段切换（planning / execution / reflection），KV cache 友好布局（稳定前缀在前，动态 history 在后）

### Prompt 模板系统（`scaffold/prompts/`）

所有 prompt 用 Jinja2 模板管理，支持变量注入和阶段分支，与代码完全解耦：

```
scaffold/prompts/
├── system/fs_agent.j2       # 主 system prompt
├── reflection/              # 循环检测时的 reflection 提示
└── compression/             # 上下文压缩摘要提示
```

### Skills Loader（`scaffold/skills/`）

读取 SKILL.md 的 YAML frontmatter（name / trigger-keywords / allowed-tools）和正文，暴露 `Skill` dataclass 供 `SkillTriggerMiddleware` 使用。触发匹配大小写不敏感，关键词任一命中即触发。

### LLM Adapter（`scaffold/models/`）

- 支持任何 **OpenAI 兼容协议**（GLM / MiniMax / DeepSeek / Moonshot 等）
- 自动处理 streaming tool call 分块拼接
- 内置指数退避重试（超时、限流、5xx 自动重试）
- `MockModel`：脚本化响应，用于 eval 离线回放，零 API 成本

### Tool Runtime（`scaffold/tools/`）

- `@tool` 装饰器：从函数类型注解和 docstring **自动生成** OpenAI function-call schema
- 同步函数自动转异步（`asyncio.to_thread`）
- 标准化错误码：`NOT_FOUND` / `PATH_OUTSIDE_SANDBOX` / `FILE_TOO_LARGE` / `UNSUPPORTED_FORMAT` / `INTERNAL_ERROR`，每种错误附带给模型的修复提示
- Pre / Post Hooks：工具执行前后可注册任意钩子（日志、审计）

### Observability（`scaffold/observability/`）

- `Tracer`：每次 Agent 运行构建 Span 树，记录 latency / token / status
- `TraceStorage`：SQLite 持久化，保存全部历史运行
- `ui.py`：Streamlit 可视化界面，展示时间线瀑布图、Token 流向、工具详情

### Cache（`scaffold/cache/`）

- TTL in-memory 缓存，以工具名 + 参数 hash 为 key
- 相同参数的工具调用命中缓存不重复执行，eval 可零成本回放
- 统计命中 / 未命中率，可用于优化工具调用频率

---

## 安全机制

### 路径沙箱

所有文件操作受 `PathSandbox` 限制，Agent 只能访问指定工作区目录。路径穿越（`../../etc/passwd`、符号链接逃逸、URL 编码绕过）全部在入口验证时拦截。

### Prompt Injection 防御

- 工具返回内容统一用 `<tool_result>` 标签包裹，内部嵌套标签用 HTML 实体转义
- System prompt 明确告知模型：`<tool_result>` 里的内容是数据而非指令
- `RedactionMiddleware` 在内容写入 context 前自动脱敏

### 三级权限控制

| 级别 | 说明 |
|------|------|
| `read_only` | 只允许读取类工具，任何写操作直接拒绝 |
| `confirm_write`（默认） | 读操作自由执行，写 / 移动 / 删除需用户在终端确认 |
| `autonomous` | 所有操作自主执行，适合受信任的批处理场景 |

---

## 配置说明

### 环境变量（`.env`）

| 变量 | 说明 |
|------|------|
| `OPENAI_API_KEY` | LLM API Key |
| `OPENAI_API_BASE` | LLM API Base URL |
| `OPENAI_MODEL_NAME` | LLM 模型名称 |
| `EMBEDDING_API_KEY` | Embedding 服务 API Key（可选，启用后支持向量检索） |
| `EMBEDDING_API_BASE` | Embedding 服务 Base URL |
| `EMBEDDING_MODEL_NAME` | Embedding 模型名称（如 `Qwen/Qwen3-Embedding-8B`） |

### 可选依赖

```bash
uv sync --all-extras              # 安装全部（推荐）
uv sync --extra fs --extra ui     # 只安装文档解析 + Web UI
```

| extras 组 | 包含 |
|-----------|------|
| `fs` | `markitdown[pdf,docx]`、`chardet` |
| `ui` | `streamlit` |
| `eval` | `tabulate` |
| `dev` | `pytest`、`pytest-asyncio`、`ruff` |

---

## 项目结构

```
scaffold-agent-harness/
│
├── scaffold/                         # 核心 Harness 框架（可复用）
│   ├── models/
│   │   ├── base.py                   # ChatModel 接口、Message / ToolCall / Usage 类型
│   │   ├── openai_compat.py          # OpenAI 兼容适配器（streaming + 重试）
│   │   └── mock.py                   # 测试用 MockModel，脚本化回放
│   ├── tools/
│   │   ├── registry.py               # ToolRegistry、@tool 装饰器、并发执行、hooks
│   │   ├── schema.py                 # 函数签名 → OpenAI schema 自动生成
│   │   └── errors.py                 # 标准化错误码
│   ├── context/
│   │   ├── budget.py                 # Token 预算分配
│   │   ├── compression.py            # SlidingWindow / SummaryWithRefs 压缩
│   │   └── window.py                 # ContextWindow，动态 prompt，KV cache 布局
│   ├── loop/
│   │   ├── react.py                  # ReAct 主循环，并发工具执行，Middleware 调度
│   │   ├── middleware.py             # StepMiddleware 协议定义
│   │   ├── checkpoint.py             # CheckpointStore（SQLite），断点续跑
│   │   └── middlewares/
│   │       ├── tool_call_limit_middleware.py
│   │       ├── redaction_middleware.py
│   │       ├── cost_tracker_middleware.py
│   │       └── skill_trigger_middleware.py
│   ├── prompts/
│   │   ├── loader.py                 # Jinja2 模板加载器
│   │   ├── system/fs_agent.j2        # 主 system prompt 模板
│   │   ├── reflection/               # 循环检测 reflection 提示
│   │   └── compression/              # 上下文压缩摘要提示
│   ├── skills/
│   │   └── loader.py                 # SKILL.md 解析器，Skill dataclass
│   ├── safety/
│   │   ├── sandbox.py                # PathSandbox，路径白名单验证
│   │   ├── injection.py              # tool_result 标签包裹 + 嵌套转义
│   │   └── redaction.py              # 敏感信息正则检测与脱敏
│   ├── observability/
│   │   ├── tracer.py                 # Span 树 Tracer
│   │   ├── storage.py                # SQLite 持久化
│   │   └── ui.py                     # Streamlit Trace 可视化界面
│   └── cache/
│       └── cache.py                  # TTL in-memory 缓存
│
├── fs_agent/                         # 文件系统 Agent（Harness 参考实现）
│   ├── agent.py                      # FSAgent / FSAgentConfig —— 统一入口
│   ├── app.py                        # Streamlit 聊天 Web UI
│   ├── cli.py                        # 命令行交互入口
│   ├── tools/
│   │   ├── file_tools.py             # 基础文件工具（9 个）
│   │   ├── doc_tools.py              # 文档处理工具（4 个，MarkItDown）
│   │   ├── advanced_tools.py         # 高级分析工具（4 个）
│   │   ├── search_tools.py           # 语义搜索工具（2 个，需 Embedding）
│   │   ├── reference_tools.py        # retrieve_reference 工具
│   │   └── skill_tools.py            # list_skills 工具
│   ├── skills/
│   │   ├── file-organize/SKILL.md    # 文件整理技能
│   │   ├── doc-qa/SKILL.md           # 文档问答技能（含向量检索策略）
│   │   └── batch-compare/SKILL.md   # 批量对比技能
│   └── policies/
│       └── permissions.py            # FSPermissionGuard 三级权限
│
├── evals/                            # 评估框架
│   ├── runner.py                     # 执行器（规则检查 + LLM-as-judge）
│   ├── cases/                        # 52+ YAML 评估用例（12 个类别）
│   └── judges/
│       ├── llm_judge.py              # LLM-as-judge 评分器
│       └── prompts.py                # 评分提示词
│
├── tests/                            # 单元测试（159 个）
├── workspace/                        # 文档库（git 忽略）
└── pyproject.toml                    # 依赖与入口点（requires Python 3.10+）
```

---

## 技术栈

| 组件 | 选型 | 说明 |
|------|------|------|
| LLM 接入 | openai Python SDK | 兼容所有 OpenAI 协议厂商 |
| 重试 | tenacity | 指数退避，处理限流和超时 |
| Token 计数 | tiktoken | cl100k_base，支持中文 fallback |
| 文档解析 | MarkItDown | 统一将 PDF / DOCX / PPTX / XLSX / HTML 转 Markdown |
| 编码检测 | chardet | 自动识别文本文件编码 |
| Embedding | OpenAI 兼容接口 | 支持 Qwen3-Embedding 等任意 OpenAI 兼容嵌入模型 |
| 向量检索 | 内存 + pickle | 余弦相似度，零外部依赖 |
| Prompt 模板 | Jinja2 | FileSystemLoader，支持热加载 |
| Trace 存储 | SQLite | 零依赖，本地持久化，与 Checkpoint 共库 |
| Web UI | Streamlit | 聊天界面 + Trace 可视化 |
| 测试 | pytest + pytest-asyncio | asyncio_mode=auto，async 测试无需额外标注 |
| 代码检查 | ruff | 快速 linter，line-length=100 |
