"""安全层 —— 沙箱、注入防护与敏感信息脱敏。"""
from scaffold.safety.sandbox import PathSandbox
from scaffold.safety.injection import sanitize_tool_result, INJECTION_DEFENSE_PROMPT
from scaffold.safety.redaction import redact_sensitive

__all__ = [
    "PathSandbox",
    "sanitize_tool_result",
    "INJECTION_DEFENSE_PROMPT",
    "redact_sensitive",
]
