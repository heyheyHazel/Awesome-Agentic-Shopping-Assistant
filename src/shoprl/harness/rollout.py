"""One episode end to end: lease an environment, run the loop, release it."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol
from uuid import uuid4

from shoprl.harness.context import ContextPolicy
from shoprl.harness.loop import AgentLoop, Emitter
from shoprl.harness.memory import Memory
from shoprl.harness.types import Trajectory


class PoolLike(Protocol):
    def acquire(self, task_id: int, session_id: str) -> Any: ...
    def release(self, lease: Any) -> None: ...


PROMPT = "完成给定的购物任务。先调用 shop_reset 获取任务，然后只使用 shop_act 与环境交互。"


class EpisodeRunner:
    """Runs episodes for one backend against one environment pool."""

    def __init__(
        self,
        backend: Any,
        pool: PoolLike,
        *,
        context_policy: ContextPolicy | None = None,
        max_turns: int = 30,
        max_tool_calls: int = 60,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        use_memory: bool = False,
        show_persona: bool = False,
        prompt: str = PROMPT,
    ):
        self.backend = backend
        self.pool = pool
        self.context_policy = context_policy or ContextPolicy()
        self.max_turns = max_turns
        self.max_tool_calls = max_tool_calls
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_memory = use_memory
        self.show_persona = show_persona
        self.prompt = prompt

    def run(self, task_id: int, *, session_id: str | None = None,
            emit: Emitter | None = None, metadata: dict[str, Any] | None = None) -> Trajectory:
        session_id = session_id or uuid4().hex
        lease = self.pool.acquire(task_id, session_id)
        memory = Memory() if self.use_memory else None
        toolkit = _toolkit(
            lease.env,
            task_id,
            session_id,
            memory,
            lambda: self.pool.release(lease),
            self.show_persona,
        )
        loop = AgentLoop(
            self.backend,
            toolkit.registry(),
            context_policy=self.context_policy,
            memory=memory or Memory(),
            max_turns=self.max_turns,
            max_tool_calls=self.max_tool_calls,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            emit=emit,
            outcome=toolkit.outcome,
        )
        try:
            trajectory = loop.run(self.prompt, task_id=task_id, metadata=metadata)
        finally:
            toolkit.close()
        trajectory.metadata.setdefault("session_id", session_id)
        return trajectory

    def run_many(self, task_ids: list[int], **kwargs) -> list[Trajectory]:
        return [self.run(task_id, **kwargs) for task_id in task_ids]


def _toolkit(
    env: Any,
    task_id: int,
    session_id: str,
    memory: Memory | None,
    on_close: Callable[[], None],
    show_persona: bool = False,
):
    from shoprl.env.tools import ShopToolkit

    return ShopToolkit(
        env, task_id, session_id, memory=memory, on_close=on_close, show_persona=show_persona
    )
