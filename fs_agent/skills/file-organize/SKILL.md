---
name: 文件整理
description: 当用户提到整理、归类、标签化或按类型分组文件时触发
trigger-keywords: [整理, 归类, 标签, tag, 分组, 按类型, 归档, 分类]
allowed-tools: list_files file_info organize_files tag_files move_file rename_file make_directory
metadata:
  version: "1.0"
---

# 文件整理技能

## 何时使用

用户明确表达希望对工作区的文件进行整理、归类或打标签，例如：
- "帮我整理一下这个目录"
- "按类型归类文件"
- "给文件打上标签"
- "把照片和文档分开"

## 不要误用的场景

- 用户只是询问文件内容（用 doc-qa 技能）
- 用户想搜索特定文件（直接用 list_files / file_info）
- 用户要对比两个文件（用 batch-compare 技能）

## 核心框架

整理任务分为三步：**探查 → 分组 → 执行**。

### 第一步：探查（Explore）
1. `list_files` 列出目标目录的所有文件（包含子目录）
2. 对结果按扩展名、命名规律做心智分组
3. 向用户确认分组方案（如有歧义）

### 第二步：分组（Plan）
根据文件类型建议目录结构，例如：
```
workspace/
  documents/   # .pdf .docx .txt .md
  data/        # .csv .json .xlsx
  code/        # .py .js .ts .sh
  media/       # .jpg .png .mp4
  archives/    # .zip .tar .gz
```
- 不要创建只有 1 个文件的分类目录（合并到 misc/）
- 如有已存在的合理目录结构，优先沿用

### 第三步：执行（Execute）
1. `make_directory` 创建目标目录
2. `move_file` 逐一移动文件
3. 完成后再次 `list_files` 验证结果
4. 向用户报告：移动了几个文件 / 创建了几个目录

## 输出结构建议

```
整理完成：
- 创建目录：documents/, data/, code/
- 移动文件：12 个
  - 5 个 PDF → documents/
  - 4 个 CSV → data/
  - 3 个 Python 脚本 → code/
```

## 示例问题

- "帮我把 workspace 里的文件按类型整理一下"
- "这些文件太乱了，能归类一下吗"
- "按扩展名分组，创建对应子目录"
