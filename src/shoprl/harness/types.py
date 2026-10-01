"""Canonical message, step and trajectory records.

The same records back three consumers, which is the point of the design:

* serving streams them to the SSE endpoint,
* training turns them into token-level samples with a loss mask,
* evaluation scores the terminal reward they carry.

A ``Message`` keeps the token ids and logprobs that produced it whenever the
backend can report them, so training never re-tokenises model output to guess
what was sampled.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Message:
    role: Role
    content: str = ""
    name: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    # Sampled tokens backing `content` + `tool_calls`, when the backend reports them.
    token_ids: list[int] | None = None
    logprobs: list[float] | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            payload["name"] = self.name
        if self.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in self.tool_calls
            ]
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        return payload


@dataclass(slots=True)
class ModelResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    finish_reason: str = "stop"


@dataclass(slots=True)
class Step:
    """One model decision plus the observations it triggered.

    ``context`` is the exact message list the model saw for this decision, after
    the context policy ran. SFT samples are built from it, so a trained turn is
    byte-identical to the turn that was rolled out.
    """

    index: int
    context: list[Message]
    assistant: Message
    observations: list[Message] = field(default_factory=list)
    latency_ms: float = 0.0


@dataclass(slots=True)
class Trajectory:
    task_id: int
    messages: list[Message] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    reward: float = 0.0
    reward_detail: dict[str, float] = field(default_factory=dict)
    termination: str = "turn_limit"
    purchase: dict[str, Any] | None = None
    goal: dict[str, Any] | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    @property
    def terminated_by_environment(self) -> bool:
        return self.termination == "done"

    @property
    def model_turns(self) -> int:
        return len(self.steps)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "task_id": self.task_id,
            "reward": self.reward,
            "reward_detail": self.reward_detail,
            "termination": self.termination,
            "purchase": self.purchase,
            "goal": self.goal,
            "error": self.error,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "messages": [message.as_dict() for message in self.messages],
            "contexts": [
                [message.as_dict() for message in step.context] for step in self.steps
            ],
            "targets": [
                {
                    "message": step.assistant.as_dict(),
                    "token_ids": step.assistant.token_ids,
                    "logprobs": step.assistant.logprobs,
                }
                for step in self.steps
            ],
        }
