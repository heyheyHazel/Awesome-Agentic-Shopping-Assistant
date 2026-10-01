"""Context management: the copy of the transcript that is sent to the model.

The policy never mutates the recorded trajectory. It returns a shortened view,
which is what makes the harness trainable: the full history stays available for
analysis, and each step keeps the exact prompt the model actually saw.

Rules, applied in order:

1. prune old tool results to a placeholder, keeping the most recent few,
2. drop whole oldest steps while the prompt exceeds the token budget,
3. never leave a tool call without its result, or a result without its call.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from shoprl.harness.types import Message

PRUNED_TOOL_RESULT = "[earlier observation pruned from context]"


def estimate_tokens(messages: Sequence[Message]) -> int:
    """Cheap language-agnostic estimate: CJK is ~1 token/char, ASCII ~4 chars/token."""
    total = 0
    for message in messages:
        text = message.content + "".join(str(call.arguments) for call in message.tool_calls)
        cjk = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
        total += cjk + (len(text) - cjk + 3) // 4
    return total


@dataclass(frozen=True)
class ContextPolicy:
    keep_recent_tool_results: int = 3
    max_prompt_tokens: int | None = None
    count_tokens: Callable[[Sequence[Message]], int] = estimate_tokens

    def apply(self, messages: Sequence[Message]) -> list[Message]:
        """Return the shortened view of ``messages`` for the next model call."""
        view = self._prune_tool_results(list(messages))
        if self.max_prompt_tokens is not None:
            view = self._drop_oldest_steps(view)
        return view

    def _prune_tool_results(self, messages: list[Message]) -> list[Message]:
        tool_indices = [i for i, message in enumerate(messages) if message.role == "tool"]
        drop = set(tool_indices[: max(0, len(tool_indices) - self.keep_recent_tool_results)])
        return [
            Message(role="tool", content=PRUNED_TOOL_RESULT, tool_call_id=message.tool_call_id)
            if index in drop
            else message
            for index, message in enumerate(messages)
        ]

    def _drop_oldest_steps(self, messages: list[Message]) -> list[Message]:
        head_end = self._head_end(messages)
        head, tail = messages[:head_end], messages[head_end:]
        # The newest step is never dropped: a prompt with no observation left
        # cannot be acted on, and training on it would teach the model to guess.
        while self._assistant_turns(tail) > 1 and self.count_tokens(head + tail) > self.max_prompt_tokens:
            tail = tail[self._span_of_oldest_step(tail) :]
        return head + tail

    @staticmethod
    def _assistant_turns(messages: Sequence[Message]) -> int:
        return sum(1 for message in messages if message.role == "assistant")

    @staticmethod
    def _head_end(messages: Sequence[Message]) -> int:
        """Pin the system prompt, the memory block and the task statement itself."""
        for index, message in enumerate(messages):
            if message.role == "user":
                return index + 1
        return len(messages)

    @staticmethod
    def _span_of_oldest_step(tail: Sequence[Message]) -> int:
        """Length of the oldest turn: its assistant message plus its observations."""
        span = 1
        while span < len(tail) and tail[span].role != "assistant":
            span += 1
        return span


@dataclass(slots=True)
class ContextStats:
    """What the policy did to one prompt, recorded per step for offline analysis."""

    tokens: int = 0
    pruned_tool_results: int = 0
    dropped_messages: int = 0
    detail: dict[str, Any] = field(default_factory=dict)
