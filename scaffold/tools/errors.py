"""Standardized tool errors with codes and hints for the model."""
from __future__ import annotations

from enum import Enum


class ToolErrorCode(str, Enum):
    INVALID_ARGUMENTS = "invalid_arguments"
    NOT_FOUND = "not_found"
    PERMISSION_DENIED = "permission_denied"
    PATH_OUTSIDE_SANDBOX = "path_outside_sandbox"
    TIMEOUT = "timeout"
    INTERNAL_ERROR = "internal_error"
    FILE_TOO_LARGE = "file_too_large"
    UNSUPPORTED_FORMAT = "unsupported_format"


# Hints the model can use to self-correct
_DEFAULT_HINTS: dict[ToolErrorCode, str] = {
    ToolErrorCode.INVALID_ARGUMENTS: "Check the required parameters and their types.",
    ToolErrorCode.NOT_FOUND: "Verify the path exists. Use list_files first to discover available files.",
    ToolErrorCode.PERMISSION_DENIED: "This operation is not allowed under the current permission level.",
    ToolErrorCode.PATH_OUTSIDE_SANDBOX: "The path must be inside the allowed workspace directory.",
    ToolErrorCode.TIMEOUT: "The operation took too long. Try a smaller scope or add filters.",
    ToolErrorCode.INTERNAL_ERROR: "An unexpected error occurred. Try again or use a different approach.",
    ToolErrorCode.FILE_TOO_LARGE: "File exceeds the size limit. Use read_file with offset/length to read chunks.",
    ToolErrorCode.UNSUPPORTED_FORMAT: "This file format is not supported. Try converting it first.",
}


class ToolError(Exception):
    """Rich error that carries a code + hint for the model."""

    def __init__(
        self,
        code: ToolErrorCode,
        message: str,
        hint: str | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.hint = hint or _DEFAULT_HINTS.get(code, "")
        super().__init__(message)

    def for_model(self) -> str:
        """Format error string intended for the LLM context."""
        parts = [f"[{self.code.value}] {self.message}"]
        if self.hint:
            parts.append(f"Hint: {self.hint}")
        return "\n".join(parts)
