"""LLM 适配器的统一 ChatModel 接口。"""
from scaffold.models.base import ChatModel, Message, ToolCall, ToolResult, ModelResponse, Usage
from scaffold.models.mock import MockModel

try:
    from scaffold.models.openai_compat import OpenAICompatModel
except ModuleNotFoundError:
    OpenAICompatModel = None

__all__ = [
    "ChatModel",
    "Message",
    "ToolCall",
    "ToolResult",
    "ModelResponse",
    "Usage",
    "MockModel",
]

if OpenAICompatModel is not None:
    __all__.append("OpenAICompatModel")
