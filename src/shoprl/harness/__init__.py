"""A framework-free tool-calling harness shared by serving and training."""

from shoprl.harness.backends import ModelBackend, OpenAIBackend, ScriptedBackend
from shoprl.harness.context import ContextPolicy
from shoprl.harness.loop import AgentLoop
from shoprl.harness.memory import Memory
from shoprl.harness.tools import ToolRegistry, ToolSpec
from shoprl.harness.types import Message, ModelResponse, Step, ToolCall, Trajectory

__all__ = [
    "AgentLoop",
    "ContextPolicy",
    "Memory",
    "Message",
    "ModelBackend",
    "ModelResponse",
    "OpenAIBackend",
    "ScriptedBackend",
    "Step",
    "ToolCall",
    "ToolRegistry",
    "ToolSpec",
    "Trajectory",
]

