"""Task pools built from the ShopSimulator release.

The upstream row order is the task id space, so a pool is a list of ids plus the
metadata needed to stratify: leaf category for coverage, option count for
difficulty. Splits are disjoint by construction and seeded, so a rerun produces
the same pools and two runs stay comparable.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shoprl.env.catalog import Catalog, load_catalog
from shoprl.settings import get_settings

TASK_DIR = "tasks"


@dataclass(slots=True)
class Task:
    task_id: int
    split: str
    tag: str
    category: str
    leaf: str
    asin: str
    instruction: str
    option_count: int
    attribute_count: int
    has_persona: bool = False

    @property
    def difficulty(self) -> int:
        return self.option_count + self.attribute_count

    def to_json(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "split": self.split,
            "tag": self.tag,
            "category": self.category,
            "leaf": self.leaf,
            "asin": self.asin,
            "instruction": self.instruction,
            "option_count": self.option_count,
            "attribute_count": self.attribute_count,
            "has_persona": self.has_persona,
        }

    @classmethod
    def from_json(cls, record: dict[str, Any]) -> Task:
        return cls(
            task_id=int(record["task_id"]),
            split=str(record.get("split") or ""),
            tag=str(record.get("tag") or ""),
            category=str(record.get("category") or ""),
            leaf=str(record.get("leaf") or ""),
            asin=str(record.get("asin") or ""),
            instruction=str(record.get("instruction") or ""),
            option_count=int(record.get("option_count") or 0),
            attribute_count=int(record.get("attribute_count") or 0),
            has_persona=bool(record.get("has_persona")),
        )


def task_from_product(product, split: str) -> Task:
    leaf = (product.category.split("›") or [""])[-1]
    return Task(
        task_id=product.index,
        split=split,
        tag=product.tag,
        category=product.category,
        leaf=leaf,
        asin=product.asin,
        instruction=product.instruction,
        option_count=len(product.instruction_options),
        attribute_count=len(product.instruction_attributes),
        has_persona=product.has_persona,
    )


def _stratified_sample(tasks: list[Task], count: int, seed: int) -> list[Task]:
    """Round-robin over categories, easiest first, so pools stay comparable."""
    if count >= len(tasks):
        return list(tasks)
    buckets: dict[str, list[Task]] = {}
    for task in sorted(tasks, key=lambda item: (item.difficulty, item.task_id)):
        buckets.setdefault(task.leaf, []).append(task)
    order = sorted(buckets)
    random.Random(seed).shuffle(order)
    picked: list[Task] = []
    while len(picked) < count and any(buckets[name] for name in order):
        for name in order:
            if buckets[name] and len(picked) < count:
                picked.append(buckets[name].pop(0))
    return sorted(picked, key=lambda item: item.task_id)


def build_task_pools(
    catalog: Catalog | None = None,
    *,
    out_dir: Path | None = None,
    sizes: dict[str, int] | None = None,
    seed: int = 20260101,
) -> dict[str, Any]:
    """Write ``official_test`` / ``dev`` / ``sft`` / ``rl`` pools and a manifest."""
    catalog = catalog or load_catalog()
    out_dir = out_dir or get_settings().data_dir / TASK_DIR
    sizes = {"dev": 150, "sft": 512, "rl": 500, **(sizes or {})}

    evaluation = [task_from_product(p, "official_test") for p in catalog if p.tag == "eval"]
    with_persona = [task for task in evaluation if task.has_persona]
    training = [task_from_product(p, "train") for p in catalog if p.tag == "train"]

    rng = random.Random(seed)
    shuffled = list(training)
    rng.shuffle(shuffled)
    dev = _stratified_sample(shuffled[: sizes["dev"] * 4], sizes["dev"], seed + 1)
    dev_ids = {task.task_id for task in dev}
    remaining = [task for task in training if task.task_id not in dev_ids]
    sft = _stratified_sample(remaining, sizes["sft"], seed + 2)
    sft_ids = {task.task_id for task in sft}
    rest = [task for task in remaining if task.task_id not in sft_ids]
    rl = _stratified_sample(rest, sizes["rl"], seed + 3)

    pools = {
        "official_test": evaluation,
        "official_test_persona": with_persona,
        "dev": dev,
        "sft": sft,
        "rl": rl,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for name, tasks in pools.items():
        path = out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for task in tasks:
                handle.write(json.dumps(task.to_json(), ensure_ascii=False) + "\n")
        counts[name] = len(tasks)

    manifest = {
        "schema_version": 1,
        "seed": seed,
        "catalogue_size": len(catalog),
        "pools": counts,
        "splits": {name: sorted({task.split for task in tasks}) for name, tasks in pools.items()},
        "categories": {name: len({task.leaf for task in tasks}) for name, tasks in pools.items()},
        "persona_tasks": {name: sum(1 for task in tasks if task.has_persona) for name, tasks in pools.items()},
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def load_task_pool(name: str, path: Path | None = None) -> list[Task]:
    path = path or get_settings().data_dir / TASK_DIR / f"{name}.jsonl"
    if not path.exists():
        build_task_pools()
    with path.open(encoding="utf-8") as handle:
        return [Task.from_json(json.loads(line)) for line in handle if line.strip()]


def pool_ids(tasks: Iterable[Task]) -> list[int]:
    return [task.task_id for task in tasks]
