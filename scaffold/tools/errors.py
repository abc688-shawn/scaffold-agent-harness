"""带有错误码和提示信息的标准化工具错误，供模型使用。"""
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


# 供模型自我修正使用的提示信息
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
    """携带错误码和提示信息的增强错误，供模型使用。"""

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
        """格式化为适合放入 LLM 上下文的错误字符串。"""
        parts = [f"[{self.code.value}] {self.message}"]
        if self.hint:
            parts.append(f"Hint: {self.hint}")
        return "\n".join(parts)
