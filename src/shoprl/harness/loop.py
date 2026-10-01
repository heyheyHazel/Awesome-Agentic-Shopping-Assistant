"""The agent loop.

One ``run`` is one episode: the harness calls the model, executes whatever tools
it asked for, appends the results, and repeats until the environment reports a
terminal state or a budget is exhausted. Everything the loop produces is
recorded, so the same object drives streaming, SFT data and RL reward.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from shoprl.harness.context import ContextPolicy, estimate_tokens
from shoprl.harness.memory import Memory
from shoprl.harness.tools import ToolRegistry
from shoprl.harness.types import Message, Step, ToolCall, Trajectory

Emitter = Callable[[dict[str, Any]], None]
OutcomeReader = Callable[[], dict[str, Any]]
TokenSink = Callable[[str], None]

# Terminal observations carry this prefix so the loop can stop without reaching
# into environment internals.
TERMINAL_MARKER = "__TERMINAL__ "

SYSTEM_PROMPT = """You are a shopping agent in an online store simulator.

Work the assigned task with the tools you are given, then finish. Rules:

1. Start by reading the task, then search before you click anything.
2. Every observation is a page. Only act on values listed under "可点击的按钮".
3. Choose the product options the task asks for before you buy; buying an
   unselected option scores as an unmet requirement.
4. Buy exactly one product, and only when it satisfies every stated requirement.
5. Never repeat an action that produced no page change.

Call tools silently. The final answer is the action you take, not prose."""


def _summary(observation: str, limit: int = 200) -> str:
    """First line of a tool result, which is the part a human reads on the card."""
    text = (observation or "").strip()
    if not text:
        return ""
    return text.splitlines()[0][:limit]


class AgentLoop:
    """Drives one episode against a tool registry, recording every decision."""

    def __init__(
        self,
        backend: Any,
        tools: ToolRegistry,
        *,
        system_prompt: str = SYSTEM_PROMPT,
        context_policy: ContextPolicy | None = None,
        memory: Memory | None = None,
        max_turns: int = 30,
        max_tool_calls: int = 60,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        emit: Emitter | None = None,
        outcome: OutcomeReader | None = None,
        on_token: TokenSink | None = None,
    ):
        self.backend = backend
        self.tools = tools
        self.system_prompt = system_prompt
        self.context_policy = context_policy or ContextPolicy()
        self.memory = memory or Memory()
        self.max_turns = max_turns
        self.max_tool_calls = max_tool_calls
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.emit = emit or (lambda _event: None)
        self.outcome = outcome
        self.on_token = on_token

    def run(
        self,
        task: str,
        *,
        task_id: int = 0,
        metadata: dict[str, Any] | None = None,
        history: list[Message] | None = None,
    ) -> Trajectory:
        """Run one episode for ``task`` and return the recorded trajectory.

        ``history`` seeds an existing transcript, which is how a multi-turn
        conversation continues without the loop owning session state.
        """
        trajectory = Trajectory(task_id=task_id, metadata=dict(metadata or {}))
        trajectory.messages.append(Message(role="system", content=self.system_prompt))
        trajectory.messages.extend(history or [])
        trajectory.messages.append(Message(role="user", content=task))

        tool_calls = 0
        termination = "turn_limit"
        while len(trajectory.steps) < self.max_turns:
            self._pin_memory(trajectory)
            context = self.context_policy.apply(trajectory.messages)
            self.emit({"type": "context", "tokens": estimate_tokens(context)})

            started = time.perf_counter()
            try:
                response = self.backend.complete(
                    context,
                    self.tools.schemas(),
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    on_token=self.on_token,
                )
            except Exception as exc:  # noqa: BLE001 - a dead backend still ends the episode
                trajectory.termination = "backend_error"
                trajectory.error = f"{type(exc).__name__}: {exc}"
                return trajectory

            assistant = Message(
                role="assistant",
                content=response.text,
                tool_calls=response.tool_calls,
                token_ids=response.token_ids or None,
                logprobs=response.logprobs or None,
            )
            step = Step(
                index=len(trajectory.steps),
                context=context,
                assistant=assistant,
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
            )
            trajectory.steps.append(step)
            trajectory.messages.append(assistant)

            if not assistant.tool_calls:
                termination = "done" if self._environment_terminated(trajectory) else "turn_limit"
                break

            for call in assistant.tool_calls:
                tool_calls += 1
                self.emit({"type": "tool", "tool": call.name, "status": "running", "message": ""})
                observation, ok = self.tools.invoke(call)
                self.emit(
                    {
                        "type": "tool",
                        "tool": call.name,
                        "status": "done" if ok else "error",
                        "message": _summary(observation),
                    }
                )
                message = Message(role="tool", content=observation, tool_call_id=call.id, name=call.name)
                step.observations.append(message)
                trajectory.messages.append(message)
                if self._environment_terminated(trajectory):
                    termination = "done"
                    break

            if termination == "done":
                break
            if tool_calls >= self.max_tool_calls:
                termination = "turn_limit"
                break

        trajectory.termination = self._final_termination(trajectory, termination)
        self._collect_outcome(trajectory)
        self.emit({"type": "trajectory", "termination": trajectory.termination, "reward": trajectory.reward})
        return trajectory

    # ── internals ─────────────────────────────────────────────────────

    def _pin_memory(self, trajectory: Trajectory) -> None:
        block = self.memory.render()
        if not block:
            return
        note = Message(role="system", content=block)
        pinned = trajectory.messages[1] if len(trajectory.messages) > 1 else None
        if pinned is not None and pinned.role == "system":
            trajectory.messages[1] = note
        else:
            trajectory.messages.insert(1, note)

    def _environment_terminated(self, trajectory: Trajectory) -> bool:
        """A trajectory is terminal once a tool reports the episode is over."""
        for message in reversed(trajectory.messages):
            if message.role == "tool" and message.content.startswith(TERMINAL_MARKER):
                return True
        return False

    def _final_termination(self, trajectory: Trajectory, fallback: str) -> str:
        return "done" if self._environment_terminated(trajectory) else fallback

    def _collect_outcome(self, trajectory: Trajectory) -> None:
        """Ask the environment for the verdict it computed when the episode ended."""
        if self.outcome is None:
            return
        verdict = self.outcome() or {}
        trajectory.reward = float(verdict.get("reward", 0.0))
        trajectory.reward_detail = dict(verdict.get("reward_detail") or {})
        trajectory.purchase = verdict.get("purchase")
        trajectory.goal = verdict.get("goal")
