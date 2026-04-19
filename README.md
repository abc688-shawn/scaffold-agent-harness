# Scaffold

**通用 Agent Harness 框架** — 以 `fs-agent` 作为参考实现。

```
┌─────────────────────────────────────────────────┐
│  应用层: fs-agent (文件系统 agent)               │
│  - 15+ 工具: 文件操作、PDF、DOCX、搜索、标签     │
│  - 策略: 沙箱、权限、脱敏                        │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│  Harness 层: Scaffold (核心)                    │
│  - Agent Loop / Tool Runtime / Context Manager  │
│  - 可观测性 / 评估 / 安全 / 缓存                 │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│  模型层: LLM Adapter                            │
│  - OpenAI 兼容 (DeepSeek, GLM, MiniMax…)        │
└─────────────────────────────────────────────────┘
```

## 快速开始

```bash
# 安装
pip install -e ".[fs,dev]"

# 运行测试（121 个测试用例）
pytest

# 启动文件 Agent（交互式 CLI）
export DEEPSEEK_API_KEY="your-key"
python -m fs_agent.cli --workspace ~/Documents --model deepseek-reasoner

# 运行评估基准（52 个用例，规则评估）
python -m evals.runner --cases evals/cases/ --dry-run

# 带 LLM-as-judge 的评估
python -m evals.runner --cases evals/cases/ --judge --output results.json

# 启动 Trace 可视化
streamlit run scaffold/observability/ui.py -- --db traces.db
```

## 架构 — 八大核心模块

| 模块 | 说明 |
|------|------|
| **LLM Adapter** | 统一 `ChatModel` 接口，OpenAI 兼容适配器，测试用 `MockModel` |
| **Tool Runtime** | `@tool` 装饰器，自动 schema 生成，错误码 + 修复提示，权限守卫，前/后钩子 |
| **Context Manager** | Token 预算管理，滑动窗口 & 摘要+引用压缩，阶段感知动态 prompt |
| **Agent Loop** | ReAct 循环，双重预算（步数 + token），循环检测 + 反思注入 |
| **可观测性** | 基于 Span 的 Trace，SQLite 持久化，Streamlit Web UI |
| **安全层** | 路径沙箱，Prompt Injection 防御，敏感信息脱敏 |
| **缓存层** | 内存缓存 + TTL，key 生成策略 |
| **评估框架** | 52+ YAML 用例，规则 + LLM-as-judge 双重评估，分类过滤 |

## fs-agent 工具集（15+）

| 工具 | 分类 | 说明 |
|------|------|------|
| `list_files` | 基础 | 列出目录内容及元数据 |
| `read_file` | 基础 | 读取文件内容（文本、代码） |
| `search_files` | 基础 | 按文件名或内容模式搜索 |
| `file_info` | 基础 | 获取文件元数据（大小、日期） |
| `write_file` | 写入 | 创建或覆盖文件 |
| `move_file` | 写入 | 移动或重命名文件 |
| `delete_file` | 写入 | 删除文件 |
| `read_pdf` | 文档 | 提取 PDF 文本（支持页码范围） |
| `read_docx` | 文档 | 提取 Word 文档文本 |
| `preview_file` | 文档 | 智能预览（自动识别格式） |
| `summarize_file` | 文档 | 结构化元数据提取 |
| `organize_files` | 高级 | 按扩展名/日期/大小整理（支持 dry-run） |
| `tag_files` | 高级 | JSON 文件标签系统 |
| `search_by_tag` | 高级 | 按标签检索文件 |
| `compare_files` | 高级 | 两文件统一 diff 对比 |
| `index_files` | 搜索 | 构建 Embedding 索引 |
| `semantic_search` | 搜索 | 向量相似度语义搜索 |

## 权限系统

三级权限，通过 `PermissionGuard` 协议强制执行：
- **read_only** — 仅允许读取类工具
- **confirm_write** — 写入类工具需要交互式确认
- **autonomous** — 所有操作无需确认

## 评估基准

52 个用例，覆盖 12 个类别：

| 类别 | 用例数 | 说明 |
|------|--------|------|
| security | 13 | 注入攻击、路径穿越、权限测试 |
| multi_step | 6 | 多工具协作流程 |
| tool_selection | 5 | 工具选择正确性 |
| edge_case | 5 | 空目录、二进制文件、缺失文件 |
| basic | 4 | 基础文件操作 |
| context | 4 | 跨文件分析、代码追踪 |
| metadata | 3 | 文件计数、大小、日期 |
| doc_processing | 3 | PDF、DOCX、预览 |
| reasoning | 3 | 设计问题、错误诊断 |
| writing | 2 | 文档生成、测试建议 |
| organization | 2 | 文件整理、标签管理 |
| error_recovery | 2 | 错误路径、不可能请求 |

## 项目结构

```
scaffold/           # 核心 harness 库（不依赖应用层）
  models/           # LLM 适配器 (base, openai_compat, mock)
  tools/            # 工具运行时 (registry, schema, errors)
  context/          # 上下文管理 (budget, compression, window)
  loop/             # Agent 循环 (react)
  safety/           # 安全层 (sandbox, injection, redaction)
  observability/    # 可观测性 (tracer, storage, ui)
  cache/            # 缓存

fs_agent/           # 文件系统 agent（参考实现）
  tools/            # 文件工具、文档工具、搜索工具、高级工具
  policies/         # 权限策略
  cli.py            # 交互式 CLI

evals/              # 评估框架
  cases/            # 52+ YAML 基准用例
  judges/           # LLM-as-judge 提示词和评分器
  runner.py         # 评估执行器（含 CLI）

tests/              # 121 个单元测试
docs/               # 设计文档和面试故事
```

## 设计原则

1. **Harness ≠ App** — `scaffold/` 永远不导入 `fs_agent/`
2. **评估驱动** — 每个优化都有基准数据支撑
3. **从第一天起可观测** — 每个 LLM/工具调用都有 Trace
4. **默认安全** — 沙箱路径、注入防御、自动脱敏

## 许可证

MIT
