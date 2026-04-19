"""Context Manager — token budgeting, message compression, dynamic prompts."""
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
