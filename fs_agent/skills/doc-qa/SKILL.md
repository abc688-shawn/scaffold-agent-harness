---
name: doc-qa
description: 当用户要求总结、摘要、解读、提问某个文档内容时触发
trigger-keywords: [总结, 概述, 摘要, 讲什么, 说了什么, 解读, 内容, 归纳, 提炼, 帮我看看, 这份文档]
allowed-tools: read_document search_document summarize_file file_info list_files
metadata:
  version: "2.0"
---

# 文档问答技能

## 何时使用

用户希望了解某个或某类文档的内容，例如：
- "这份 PDF 讲了什么"
- "帮我总结一下这几个文档"
- "report.pdf 里关于利润的部分在哪"
- "这个文件的主要内容是什么"

## 不要误用的场景

- 用户想整理文件结构 → 用 file-organize 技能
- 用户想对比两个文件的差异 → 用 batch-compare 技能
- 用户只想知道文件是否存在 → 直接用 list_files / file_info

## 核心框架：定位 → 阅读/检索 → 回答

### 第一步：定位（Locate）
1. 用户指定了文件名 → `file_info` 确认存在并获取大小
2. 未指定 → `list_files` 找候选文件，结合上下文判断目标
3. 多个候选文件 → 用 `summarize_file` 快速获取结构概览，选定目标

### 第二步：阅读或检索（Read / Search）

**根据任务类型选择工具**：

| 场景 | 推荐工具 |
|---|---|
| 总结整份文档、了解全文结构 | `read_document` |
| 文档 < 10KB | `read_document` |
| 文档较大，用户问的是具体问题 | `search_document` |
| 用户追问某个细节 | `search_document` |

- `read_document(path)` — 用 MarkItDown 将 PDF/DOCX/PPTX/XLSX 转为 Markdown 全文返回
- `search_document(path, query)` — 对文档分块建向量索引，只返回与 query 最相关的段落（首次调用建索引，后续复用缓存）

**优先使用 `search_document` 的场景**：
- 用户问的是文档中某个具体内容（"第三季度利润"、"风险章节"）
- 文档超过 20KB，读全文会消耗大量上下文
- 用户多次追问同一文档的不同细节

### 第三步：回答（Answer）
- 先给出**一句话摘要**，再展开**要点列表**
- 引用文档原文片段作为依据（加引号标注）
- 如 `search_document` 返回的段落不足以回答，再调用 `read_document` 获取更多上下文
- 信息不足时，明确说明并建议用户进一步指定范围

## 输出结构建议

```
**一句话摘要**：本文档介绍了…

**主要内容**：
1. …
2. …

**关键数据 / 结论**：
- …（引用自原文："…"）
```

## 示例问题

- "report.pdf 讲了什么，帮我总结"
- "这份合同里关于违约条款是怎么说的"
- "meeting_notes.docx 的主要结论是什么"
- "帮我看看这个目录下几个文档的主要内容"
