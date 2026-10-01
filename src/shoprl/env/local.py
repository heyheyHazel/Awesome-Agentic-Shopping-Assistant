"""In-process ShopSimulator environment and the rollout env pool.

The service contract matches the upstream Flask API (``reset`` / ``interact`` /
``release``) so the harness can be pointed at either implementation without
changing a line of agent code. The pool exists because rollouts run concurrently:
one environment per in-flight episode, never a shared one, or states leak between
tasks and the reward stops meaning anything.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from shoprl.env.actions import MAX_HISTORY_LENGTH
from shoprl.env.catalog import Catalog, Product
from shoprl.env.search import Bm25Index
from shoprl.env.session import ShopSession


class LocalShopEnv:
    """One environment slot. Not thread-safe on its own; the pool locks it."""

    def __init__(self, catalog: Catalog, index: Bm25Index, *, env_idx: int = 0,
                 history_limit: int = MAX_HISTORY_LENGTH):
        self.catalog = catalog
        self.index = index
        self.env_idx = env_idx
        self.history_limit = history_limit
        self.session: ShopSession | None = None
        self.session_id: str = ""
        self.task_id: int | None = None

    @property
    def busy(self) -> bool:
        return self.session is not None

    def reset(self, task_id: int, session_id: str) -> dict[str, Any]:
        product: Product | None = self.catalog.at(task_id)
        if product is None:
            raise ValueError(f"task {task_id} is outside the catalogue (0..{len(self.catalog) - 1})")
        self.session = ShopSession(
            self.catalog, self.index, product, history_limit=self.history_limit
        )
        self.session_id, self.task_id = session_id, task_id
        self.session.reset()
        return {
            "message": f"Task {task_id} started",
            "instruction": product.instruction,
            "instruction_simple": product.instruction_simple,
            "goal_options": list(product.instruction_options),
            "user_persona": product.persona,
            "env_idx": self.env_idx,
            "idx": task_id,
            "task_id": task_id,
            "rollout_session_id": session_id,
        }

    def interact(self, action: str, session_id: str) -> dict[str, Any]:
        if self.session is None or self.session_id != session_id:
            raise ValueError(
                f"environment {self.env_idx} has no session {session_id!r} "
                f"(active {self.session_id!r})"
            )
        outcome = self.session.step(action)
        return {
            "message": "Continue interaction",
            "instruction": outcome.text,
            "done": outcome.done,
            "reward": outcome.reward,
            "reward_detail": outcome.reward_detail,
            "purchase": outcome.purchase or {},
            "goal": outcome.goal or {},
            "over": outcome.over,
            "changed": outcome.changed,
            "env_idx": self.env_idx,
            "idx": self.task_id,
            "task_id": self.task_id,
            "rollout_session_id": session_id,
        }

    def release(self, session_id: str | None = None) -> None:
        if session_id is not None and self.session_id != session_id:
            raise ValueError(f"environment {self.env_idx} holds {self.session_id!r}, not {session_id!r}")
        self.session = None
        self.session_id, self.task_id = "", None


@dataclass(slots=True)
class Lease:
    env: LocalShopEnv
    session_id: str


class EnvPool:
    """Fixed-size pool of environments, allocated per rollout and always released."""

    def __init__(self, catalog: Catalog, index: Bm25Index, *, capacity: int = 8,
                 history_limit: int = MAX_HISTORY_LENGTH):
        self.capacity = capacity
        self._envs = [
            LocalShopEnv(catalog, index, env_idx=index_, history_limit=history_limit)
            for index_ in range(capacity)
        ]
        self._free = set(range(capacity))
        self._lock = threading.Lock()

    def acquire(self, task_id: int, session_id: str) -> Lease:
        with self._lock:
            if not self._free:
                raise RuntimeError(f"all {self.capacity} environments are busy")
            env = self._envs[min(self._free)]
            self._free.discard(env.env_idx)
        env.reset(task_id, session_id)
        return Lease(env=env, session_id=session_id)

    def release(self, lease: Lease) -> None:
        with self._lock:
            if lease.env.env_idx in self._free:
                return
            self._free.add(lease.env.env_idx)
        lease.env.release(lease.session_id)

    def status(self) -> dict[str, Any]:
        with self._lock:
            active = {env.env_idx: env.session_id for env in self._envs if env.busy}
            return {
                "capacity": self.capacity,
                "free": len(self._free),
                "active": len(active),
                "active_sessions": dict(sorted(active.items())),
            }
