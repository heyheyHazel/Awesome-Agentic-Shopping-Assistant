"""The metrics ShopSimulator reports, computed the same way.

``r_loose`` is the environment's scalar reward, ``r_hard`` is the product of the
four sub-scores, ``r_success`` requires all four to be exactly 1. They are kept
apart because a run that improves ``r_loose`` by buying plausible products
without the requested options is not the same result as one that satisfies every
constraint, and averaging them together hides that.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from shoprl.env.reward import SUB_SCORES


def outcome(record: dict[str, Any]) -> dict[str, Any]:
    detail = record.get("reward_detail") or {}
    scored = bool(detail)
    values = {
        name: float(detail.get(name, 1.0 if name == "r_option" else 0.0)) if scored else 0.0
        for name in SUB_SCORES
    }
    hard = 1.0
    for value in values.values():
        hard *= value
    purchase = (record.get("purchase") or {}).get("asin")
    goal = (record.get("goal") or {}).get("asin")
    return {
        "task_id": record.get("task_id"),
        "termination": record.get("termination"),
        "done": 1 if scored else 0,
        "r_loose": float(record.get("reward") or 0.0),
        "r_hard": round(hard, 6),
        "r_success": 1 if scored and all(value == 1 for value in values.values()) else 0,
        "right_product": 1 if scored and purchase and purchase == goal else 0,
        "sub_scores": values,
    }


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def summarise(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Mean scores plus pass@k statistics grouped by task."""
    outcomes = [outcome(record) for record in records]
    if not outcomes:
        return {"samples": 0}

    metrics: dict[str, Any] = {
        "samples": len(outcomes),
        "done_rate": _mean([item["done"] for item in outcomes]),
        "r_loose": _mean([item["r_loose"] for item in outcomes]),
        "r_hard": _mean([item["r_hard"] for item in outcomes]),
        "r_success": _mean([item["r_success"] for item in outcomes]),
        "right_product_rate": _mean([item["right_product"] for item in outcomes]),
        **{
            name: _mean([item["sub_scores"][name] for item in outcomes])
            for name in SUB_SCORES
        },
        "terminations": _count(outcomes, "termination"),
    }

    if any(item["task_id"] is None for item in outcomes):
        metrics["pass_at_k"] = None
        return metrics

    by_task: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for item in outcomes:
        by_task[item["task_id"]].append(item)
    counts = {len(group) for group in by_task.values()}
    metrics["pass_at_k"] = {
        "tasks": len(by_task),
        "samples_per_task": min(counts) if len(counts) == 1 else {"min": min(counts), "max": max(counts)},
        "reward_variance_task_fraction": _mean(
            [
                1.0
                if max(item["r_loose"] for item in group) != min(item["r_loose"] for item in group)
                else 0.0
                for group in by_task.values()
            ]
        ),
        "all_zero_reward_task_fraction": _mean(
            [1.0 if max(item["r_loose"] for item in group) == 0 else 0.0 for group in by_task.values()]
        ),
        "pass_success": _mean([1.0 if any(item["r_success"] for item in group) else 0.0 for group in by_task.values()]),
        "pass_positive_reward": _mean(
            [1.0 if any(item["r_loose"] > 0 for item in group) else 0.0 for group in by_task.values()]
        ),
        "pass_right_product": _mean(
            [1.0 if any(item["right_product"] for item in group) else 0.0 for group in by_task.values()]
        ),
    }
    return metrics


def _count(outcomes: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in outcomes:
        name = str(item.get(key))
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))

