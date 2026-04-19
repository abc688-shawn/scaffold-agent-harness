# 架构与设计决策

## 三层架构

系统严格分为三层：

1. **模型层** (`scaffold/models/`) — 在统一的 `ChatModel` 接口后面抽象 LLM 提供商
2. **Harness 层** (`scaffold/`) — 可复用的 Agent 基础设施
3. **应用层** (`fs_agent/`) — 特定领域的工具和策略

**核心规则**：`scaffold/` 永远不导入 `fs_agent/`，依赖方向是单向的。

## LLM Adapter 设计

选择 OpenAI 兼容适配器作为主要实现，原因：
- DeepSeek、GLM（智谱）、MiniMax、Moonshot 等均暴露 OpenAI 风格的 API
- `openai` Python SDK 统一处理流式响应、认证、错误类型
- 我们仍然定义自己的 `Message`、`ToolCall`、`ModelResponse` 类型以避免耦合

`MockModel` 实现无需 API 调用的确定性测试 —— 脚本化的响应驱动零成本 eval 回放。

## Tool Runtime

`@tool` 装饰器通过 Python 类型内省自动生成 OpenAI function-call schema：
- 添加新工具 = 写一个函数 + 一个装饰器
- Schema 与代码始终保持同步
- 错误处理标准化（每个错误都有错误码 + 给模型的修复提示）

**Permission Guard**（权限守卫）：基于 Protocol 的系统在每次工具调用前进行检查：
- `check()` 返回 `True`（允许）、`False`（拒绝）或 `"confirm"`（询问用户）
- 前/后钩子支持日志、指标和审计追踪

## Context Manager

三种压缩策略：
- **滑动窗口**：保留最近 N 条消息，将其余消息摘要化
- **摘要 + 引用**：摘要旧消息，通过引用 ID 存储工具结果以便后续检索
- **动态 Prompt**：阶段感知的 system prompt（规划 / 执行 / 反思），布局对 KV cache 友好

布局顺序：system prompt（稳定前缀）→ history（动态部分） —— 最大化跨轮次的 prefix cache 命中率。

## 安全层

三重防御：
1. **路径沙箱** — 白名单目录，解析符号链接
2. **Prompt Injection 防御** — `<tool_result>` 标签包裹 + 嵌套标签转义 + system prompt 明确拒绝
3. **敏感数据脱敏** — 正则检测 API key、邮箱、身份证号、手机号 → 自动遮罩

## 评估框架

两种评估模式：
1. **规则检查**：子串匹配、禁止内容检查、工具使用验证 —— 快速、免费、确定性
2. **LLM-as-judge**：4 个评分维度（正确性、工具选择、安全性、效率）+ 专用安全评审

52+ 用例覆盖 12 个类别，CLI 支持按类别/标签过滤，JSON 输出用于追踪回归。

## 可观测性

基于 Span 的 Trace，建模为树状结构：
- Agent 运行 → LLM 调用 → 工具执行
- 每个 Span 记录延迟、Token 用量、元数据
- SQLite 持久化存储，支持事后分析
- Streamlit Web UI：时间线瀑布图、Token 流向图、工具详情展开
