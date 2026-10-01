"""Roll one policy out over a task pool and score it."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shoprl.env.tasks import load_task_pool
from shoprl.eval.metrics import summarise
from shoprl.harness.rollout import EpisodeRunner


@dataclass(slots=True)
class EvalConfig:
    output_dir: Path
    tasks_name: str = "official_test"
    tasks: Path | None = None
    samples_per_task: int = 1
    limit: int = 0
    concurrency: int = 4
    temperature: float = 0.0
    label: str = "policy"


def evaluate(runner: EpisodeRunner, config: EvalConfig) -> dict[str, Any]:
    started = time.perf_counter()
    tasks = load_task_pool(config.tasks_name, config.tasks)
    if config.limit:
        tasks = tasks[: config.limit]
    jobs = [(task.task_id, sample) for task in tasks for sample in range(config.samples_per_task)]

    def run(job: tuple[int, int]):
        task_id, sample = job
        return runner.run(
            task_id,
            session_id=f"eval-{task_id}-{sample}",
            metadata={"sample": sample, "label": config.label},
        )

    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, config.concurrency)) as pool:
        for trajectory in pool.map(run, jobs):
            records.append(trajectory.to_json())

    config.output_dir.mkdir(parents=True, exist_ok=True)
    with (config.output_dir / "rollouts.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    metrics = {
        "label": config.label,
        "tasks": len(tasks),
        "samples_per_task": config.samples_per_task,
        "temperature": config.temperature,
        "wall_seconds": round(time.perf_counter() - started, 2),
        **summarise(records),
    }
    (config.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metrics

