"""敏感信息检测与脱敏。

当前检测：API Key、密码、中国身份证号、邮箱、手机号。
基于正则表达式实现，后续可扩展为 ML 分类器。
"""
from __future__ import annotations

import re
from typing import NamedTuple


class Detection(NamedTuple):
    kind: str
    value: str
    start: int
    end: int


# 模式按“从最具体到最宽泛”排序
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("api_key", re.compile(
        r"(?:sk|key|token|api[_-]?key)[_-]?[=:\s]+['\"]?([a-zA-Z0-9_\-]{20,})['\"]?",
        re.IGNORECASE,
    )),
    ("chinese_id", re.compile(
        r"\b(\d{17}[\dXx])\b",
    )),
    ("email", re.compile(
        r"\b([a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})\b",
    )),
    ("phone_cn", re.compile(
        r"\b(1[3-9]\d{9})\b",
    )),
    ("password_field", re.compile(
        r"(?:password|passwd|pwd)[_\s]*[=:\s]+['\"]?(\S{6,})['\"]?",
        re.IGNORECASE,
    )),
]


def detect_sensitive(text: str) -> list[Detection]:
    """扫描 *text* 中的敏感模式，并返回 Detection 列表。"""
    detections: list[Detection] = []
    for kind, pattern in _PATTERNS:
        for m in pattern.finditer(text):
            g = m.group(1) if m.lastindex else m.group(0)
            detections.append(Detection(kind=kind, value=g, start=m.start(), end=m.end()))
    return detections


def redact_sensitive(text: str, placeholder: str = "***REDACTED***") -> str:
    """将检测到的敏感信息替换为 *placeholder*。"""
    detections = detect_sensitive(text)
    if not detections:
        return text
    # 按起始位置倒序排序，确保从后往前替换
    detections.sort(key=lambda d: d.start, reverse=True)
    result = text
    for d in detections:
        result = result[:d.start] + placeholder + result[d.end:]
    return result
