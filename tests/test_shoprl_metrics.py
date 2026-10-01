"""The official metrics, including how a failed episode is counted.

The distinction these tests protect: an episode that never bought anything must
lower the mean but must not be silently dropped, because dropping it turns a
policy that gives up into a policy with a good average.
"""

from __future__ import annotations

from shoprl.eval.metrics import outcome, summarise


def record(task_id: int, reward: float = 0.0, detail: dict | None = None, **extra) -> dict:
    return {
        "task_id": task_id,
        "reward": reward,
        "reward_detail": detail or {},
        "termination": extra.pop("termination", "done" if detail else "turn_limit"),
        "purchase": extra.pop("purchase", {"asin": "A1"} if detail else None),
        "goal": extra.pop("goal", {"asin": "A1"}),
        **extra,
    }


PERFECT = {"r_type": 1.0, "r_att": 1.0, "r_option": 1.0, "r_price": 1.0}
HARD_ONLY = {"r_type": 1.0, "r_att": 0.5, "r_option": 1.0, "r_price": 1.0}


def test_a_perfect_episode_scores_one_on_every_metric():
    item = outcome(record(1, reward=1.0, detail=PERFECT))

    assert item["done"] == 1
    assert item["r_hard"] == 1.0
    assert item["r_success"] == 1
    assert item["right_product"] == 1


def test_a_loose_win_is_not_a_strict_win():
    item = outcome(record(1, reward=0.8, detail=HARD_ONLY))

    assert item["r_hard"] == 0.5
    assert item["r_success"] == 0
    assert item["done"] == 1


def test_missing_r_option_defaults_to_met_because_no_option_was_requested():
    """Upstream scores an option-less task as satisfied, and that is reproduced."""
    detail = {"r_type": 1.0, "r_att": 1.0, "r_price": 1.0}
    item = outcome(record(1, reward=1.0, detail=detail))

    assert item["sub_scores"]["r_option"] == 1.0
    assert item["r_hard"] == 1.0


def test_an_episode_that_never_finished_counts_against_the_average():
    metrics = summarise([record(1, reward=1.0, detail=PERFECT), record(2)])

    assert metrics["samples"] == 2
    assert metrics["done_rate"] == 0.5
    assert metrics["r_loose"] == 0.5
    assert metrics["r_hard"] == 0.5
    assert metrics["terminations"] == {"done": 1, "turn_limit": 1}


def test_pass_at_k_is_computed_per_task():
    metrics = summarise(
        [
            record(1, reward=1.0, detail=PERFECT),
            record(1, reward=0.0),
            record(2, reward=0.0),
            record(2, reward=0.0),
        ]
    )

    assert metrics["pass_at_k"]["tasks"] == 2
    assert metrics["pass_at_k"]["samples_per_task"] == 2
    assert metrics["pass_at_k"]["pass_success"] == 0.5
    assert metrics["pass_at_k"]["all_zero_reward_task_fraction"] == 0.5
    assert metrics["pass_at_k"]["reward_variance_task_fraction"] == 0.5


def test_pass_at_k_is_omitted_when_the_task_id_is_unknown():
    metrics = summarise([{**record(1, reward=1.0, detail=PERFECT), "task_id": None}])
    assert metrics["pass_at_k"] is None


def test_no_samples_is_not_a_division_by_zero():
    assert summarise([]) == {"samples": 0}

