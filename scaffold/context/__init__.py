"""上下文管理器 —— token 预算、消息压缩与动态提示词。"""
from scaffold.context.budget import TokenBudget
from scaffold.context.compression import compress_messages, CompressionStrategy
from scaffold.context.window import ContextWindow, DynamicPrompt, AgentPhase

__all__ = [
    "TokenBudget",
    "compress_messages",
    "CompressionStrategy",
    "ContextWindow",
    "DynamicPrompt",
    "AgentPhase",
]
