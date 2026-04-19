"""Sensitive information detection and redaction.

Detects: API keys, passwords, Chinese ID numbers, emails, phone numbers.
Uses regex patterns. Can be extended with ML classifiers later.
"""
from __future__ import annotations

import re
from typing import NamedTuple


class Detection(NamedTuple):
    kind: str
    value: str
    start: int
    end: int


# Patterns ordered from most to least specific
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
    """Scan *text* for sensitive patterns. Returns list of Detection."""
    detections: list[Detection] = []
    for kind, pattern in _PATTERNS:
        for m in pattern.finditer(text):
            g = m.group(1) if m.lastindex else m.group(0)
            detections.append(Detection(kind=kind, value=g, start=m.start(), end=m.end()))
    return detections


def redact_sensitive(text: str, placeholder: str = "***REDACTED***") -> str:
    """Replace detected sensitive info with *placeholder*."""
    detections = detect_sensitive(text)
    if not detections:
        return text
    # Sort detections by start position (reverse) to replace from end
    detections.sort(key=lambda d: d.start, reverse=True)
    result = text
    for d in detections:
        result = result[:d.start] + placeholder + result[d.end:]
    return result
