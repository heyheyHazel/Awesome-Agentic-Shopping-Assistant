"""Teacher rollout collection.

The teacher only has to be good enough to finish the task; a trajectory is kept
when the environment reports a terminal purchase, and anything else is recorded
with the reason it was dropped. Rejections are kept because the failure mix is
the first thing worth looking at when SFT underperforms.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from shoprl.env.tasks import Task
from shoprl.harness.rollout import EpisodeRunner
from shoprl.harness.types import Trajectory


@dataclass(slots=True)
class CollectionSummary:
    tasks: int = 0
    attempted: int = 0
    accepted: int = 0
    rejected: int = 0
    reasons: dict[str, int] = field(default_factory=dict)
    mean_turns: float = 0.0
    mean_reward: float = 0.0
    model_turns: float = 0.0
    wall_seconds: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "tasks": self.tasks,
            "attempted": self.attempted,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "acceptance_rate": round(self.accepted / self.attempted, 4) if self.attempted else 0.0,
            "reasons": dict(sorted(self.reasons.items())),
            "mean_model_turns": round(self.model_turns, 3),
            "mean_reward": round(self.mean_reward, 6),
            "wall_seconds": round(self.wall_seconds, 2),
        }


def acceptance(trajectory: Trajectory, *, min_reward: float = 0.0) -> tuple[bool, str]:
    """A trajectory is teacher data only when the episode actually finished."""
    if trajectory.termination == "backend_error":
        return False, "backend_error"
    if trajectory.termination != "done":
        return False, trajectory.termination
    if trajectory.reward <= min_reward:
        return False, "reward_below_threshold"
    return True, ""


def collect_trajectories(
    tasks: list[Task],
    runner: EpisodeRunner,
    *,
    samples_per_task: int = 1,
    concurrency: int = 4,
    min_reward: float = 0.0,
    on_trajectory: Callable[[Trajectory], None] | None = None,
) -> tuple[list[Trajectory], CollectionSummary]:
    """Roll the teacher out over ``tasks``; returns accepted trajectories and stats."""
    started = time.perf_counter()
    jobs = [(task.task_id, sample) for task in tasks for sample in range(samples_per_task)]
    summary = CollectionSummary(tasks=len(tasks), attempted=len(jobs))
    accepted: list[Trajectory] = []
    rewards: list[float] = []
    turns: list[int] = []

    def run(job: tuple[int, int]) -> Trajectory:
        task_id, sample = job
        return runner.run(task_id, metadata={"sample": sample})

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        for trajectory in pool.map(run, jobs):
            ok, reason = acceptance(trajectory, min_reward=min_reward)
            trajectory.metadata["accepted"] = ok
            trajectory.metadata["reject_reason"] = reason
            rewards.append(trajectory.reward)
            turns.append(trajectory.model_turns)
            if ok:
                accepted.append(trajectory)
            else:
                summary.reasons[reason] = summary.reasons.get(reason, 0) + 1
            if on_trajectory is not None:
                on_trajectory(trajectory)

    summary.accepted = len(accepted)
    summary.rejected = summary.attempted - summary.accepted
    summary.mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
    summary.model_turns = sum(turns) / len(turns) if turns else 0.0
    summary.wall_seconds = time.perf_counter() - started
    return accepted, summary


class TrajectoryWriter:
    """Append-only JSONL sink, keyed by task so a rerun can skip finished work."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, task_id: int, sample: int = 0) -> Path:
        return self.directory / f"{task_id:06d}-{sample:03d}.json"

    def write(self, trajectory: Trajectory) -> Path:
        path = self.path_for(trajectory.task_id, int(trajectory.metadata.get("sample", 0)))
        path.write_text(
            json.dumps(trajectory.to_json(), ensure_ascii=False), encoding="utf-8"
        )
        return path

    def existing_ids(self) -> set[int]:
        return {int(path.stem.split("-")[0]) for path in self.directory.glob("*.json")}


def load_trajectories(directory: Path) -> list[dict[str, Any]]:
    """Read raw trajectory records written by :class:`TrajectoryWriter`."""
    records = []
    for path in sorted(directory.glob("*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return records

