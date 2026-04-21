"""路径沙箱 —— 将文件操作限制在允许的目录内。"""
from __future__ import annotations

from pathlib import Path

from scaffold.tools.errors import ToolError, ToolErrorCode


class PathSandbox:
    """将文件系统路径限制在白名单目录之内。"""

    def __init__(self, allowed_roots: list[str | Path]) -> None:
        self._roots = [Path(r).resolve() for r in allowed_roots]

    def validate(self, path: str | Path) -> Path:
        """解析 *path*，并验证它是否位于允许的根目录内。

        成功时返回解析后的 Path。
        如果违反约束则抛出 ToolError。
        """
        resolved = Path(path).resolve()
        for root in self._roots:
            try:
                resolved.relative_to(root)
                return resolved
            except ValueError:
                continue
        raise ToolError(
            ToolErrorCode.PATH_OUTSIDE_SANDBOX,
            f"Path '{path}' is outside the allowed directories: {self._roots}",
        )

    def add_root(self, root: str | Path) -> None:
        self._roots.append(Path(root).resolve())

    @property
    def roots(self) -> list[Path]:
        return list(self._roots)
