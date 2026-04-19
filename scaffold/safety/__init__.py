"""Safety Layer — sandbox, injection defense, redaction."""
from scaffold.safety.sandbox import PathSandbox
from scaffold.safety.injection import sanitize_tool_result, INJECTION_DEFENSE_PROMPT
from scaffold.safety.redaction import redact_sensitive

__all__ = [
    "PathSandbox",
    "sanitize_tool_result",
    "INJECTION_DEFENSE_PROMPT",
    "redact_sensitive",
]
