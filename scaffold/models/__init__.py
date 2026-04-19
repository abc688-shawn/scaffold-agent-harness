"""LLM Adapter — unified ChatModel interface."""
from scaffold.models.base import ChatModel, Message, ToolCall, ToolResult, ModelResponse, Usage
from scaffold.models.openai_compat import OpenAICompatModel
from scaffold.models.mock import MockModel

__all__ = [
    "ChatModel",
    "Message",
    "ToolCall",
    "ToolResult",
    "ModelResponse",
    "Usage",
    "OpenAICompatModel",
    "MockModel",
]
