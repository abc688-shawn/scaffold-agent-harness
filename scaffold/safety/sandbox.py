"""Path sandbox — confine file operations to allowed directories."""
from __future__ import annotations

from pathlib import Path

from scaffold.tools.errors import ToolError, ToolErrorCode


class PathSandbox:
    """Restricts file-system paths to a whitelist of directories."""

    def __init__(self, allowed_roots: list[str | Path]) -> None:
        self._roots = [Path(r).resolve() for r in allowed_roots]

    def validate(self, path: str | Path) -> Path:
        """Resolve *path* and verify it falls inside an allowed root.

        Returns the resolved Path on success.
        Raises ToolError on violation.
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
