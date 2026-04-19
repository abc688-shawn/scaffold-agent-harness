"""Auto-generate OpenAI function-call schema from Python type hints."""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, get_type_hints


_PY_TO_JSON: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


@dataclass
class ToolSchema:
    """Holds the OpenAI-style function schema for one tool."""
    name: str
    description: str
    parameters: dict[str, Any]

    def to_openai(self) -> dict[str, Any]:
        """Return the dict expected by the OpenAI `tools` parameter."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def schema_from_function(fn: Callable[..., Any], name: str | None = None,
                          description: str | None = None) -> ToolSchema:
    """Inspect *fn* and produce a ToolSchema.

    - Uses type hints for parameter types.
    - Uses the docstring (first line) as description if not provided.
    - Parameters with defaults are not ``required``.
    """
    sig = inspect.signature(fn)
    hints = get_type_hints(fn)

    properties: dict[str, Any] = {}
    required: list[str] = []

    for pname, param in sig.parameters.items():
        if pname in ("self", "cls"):
            continue
        ptype = hints.get(pname, str)
        # Unwrap Optional
        origin = getattr(ptype, "__origin__", None)
        if origin is type(None):
            continue
        json_type = _PY_TO_JSON.get(ptype, "string")
        properties[pname] = {"type": json_type}

        # Parse per-param description from docstring
        doc_desc = _extract_param_doc(fn, pname)
        if doc_desc:
            properties[pname]["description"] = doc_desc

        if param.default is inspect.Parameter.empty:
            required.append(pname)

    tool_desc = description or (inspect.getdoc(fn) or "").split("\n")[0]
    tool_name = name or fn.__name__

    return ToolSchema(
        name=tool_name,
        description=tool_desc,
        parameters={
            "type": "object",
            "properties": properties,
            "required": required,
        },
    )


def _extract_param_doc(fn: Callable[..., Any], param_name: str) -> str | None:
    """Extract a parameter description from Google/Numpy-style docstring."""
    doc = inspect.getdoc(fn)
    if not doc:
        return None
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped.startswith(param_name):
            # "path: The file path to read" or "path (str): ..."
            parts = stripped.split(":", 1)
            if len(parts) == 2:
                return parts[1].strip()
    return None
