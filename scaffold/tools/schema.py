"""根据 Python 类型提示自动生成 OpenAI function-call schema。"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Union, get_args, get_origin, get_type_hints

try:
    from types import UnionType
except ImportError:  # pragma: no cover  # Python < 3.10 的兼容分支
    UnionType = None


_PY_TO_JSON: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _json_schema_type_for_annotation(annotation: Any) -> tuple[str, bool]:
    """将 Python 注解解析为 JSON Schema 类型，并标记是否允许为 null。"""
    origin = get_origin(annotation)
    args = get_args(annotation)
    union_origins = {Union}
    if UnionType is not None:
        union_origins.add(UnionType)

    # 处理 Optional[T] / T | None
    if origin in union_origins:
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1 and len(non_none_args) != len(args):
            json_type, _ = _json_schema_type_for_annotation(non_none_args[0])
            return json_type, True

    if origin is not None:
        return _PY_TO_JSON.get(origin, "string"), False

    return _PY_TO_JSON.get(annotation, "string"), False


@dataclass
class ToolSchema:
    """保存单个工具的 OpenAI 风格函数 schema。"""
    name: str
    description: str
    parameters: dict[str, Any]

    def to_openai(self) -> dict[str, Any]:
        """返回符合 OpenAI `tools` 参数要求的字典。"""
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
    """检查 *fn* 并生成一个 ToolSchema。

    - 使用类型提示推断参数类型。
    - 如果未显式提供描述，则使用 docstring 的第一行作为描述。
    - 带默认值的参数不会被标记为 ``required``。
    """
    sig = inspect.signature(fn)
    hints = get_type_hints(fn)

    properties: dict[str, Any] = {}
    required: list[str] = []

    for pname, param in sig.parameters.items():
        if pname in ("self", "cls"):
            continue
        ptype = hints.get(pname, str)
        json_type, nullable = _json_schema_type_for_annotation(ptype)
        properties[pname] = {"type": json_type}
        if nullable:
            properties[pname]["nullable"] = True

        # 从 docstring 中解析每个参数的描述
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
    """从 Google/Numpy 风格的 docstring 中提取参数描述。"""
    doc = inspect.getdoc(fn)
    if not doc:
        return None
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped.startswith(param_name):
            # 例如 "path: 要读取的文件路径" 或 "path (str): ..."
            parts = stripped.split(":", 1)
            if len(parts) == 2:
                return parts[1].strip()
    return None
