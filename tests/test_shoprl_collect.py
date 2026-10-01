"""Teacher collection: what is accepted, what is rejected, and why."""

from __future__ import annotations

from shoprl.data.collect import TrajectoryWriter, acceptance, collect_trajectories
from shoprl.env.catalog import Catalog, Product
from shoprl.env.local import EnvPool
from shoprl.env.search import Bm25Index
from shoprl.env.tasks import Task
from shoprl.harness.backends import ScriptedBackend
from shoprl.harness.rollout import EpisodeRunner
from shoprl.harness.types import ModelResponse, ToolCall, Trajectory


def build_pool() -> EnvPool:
    products = [
        Product(
            index=index,
            asin=f"A{index:04d}",
            title=f"商品 {index}",
            shop_name="店铺",
            category="服饰›上衣›卫衣",
            domain="服饰",
            attributes=["纯棉"],
            pricing=[100.0 + index],
            instruction=f"买商品 {index}",
            instruction_attributes=["纯棉"],
            instruction_options=["黑色"],
        )
        for index in range(3)
    ]
    catalog = Catalog(products)
    return EnvPool(catalog, Bm25Index.build(catalog), capacity=2)


def tasks(count: int = 3) -> list[Task]:
    return [
        Task(
            task_id=index,
            split="sft",
            tag="train",
            category="服饰›上衣›卫衣",
            leaf="卫衣",
            asin=f"A{index:04d}",
            instruction=f"买商品 {index}",
            option_count=1,
            attribute_count=1,
        )
        for index in range(count)
    ]


def trajectory(task_id: int, *, termination: str, reward: float) -> Trajectory:
    return Trajectory(task_id=task_id, termination=termination, reward=reward)


def test_only_a_finished_episode_is_teacher_data():
    assert acceptance(trajectory(1, termination="done", reward=0.4)) == (True, "")
    assert acceptance(trajectory(1, termination="turn_limit", reward=1.0)) == (False, "turn_limit")
    assert acceptance(trajectory(1, termination="backend_error", reward=0.0)) == (False, "backend_error")
    assert acceptance(trajectory(1, termination="done", reward=0.0)) == (False, "reward_below_threshold")


def solve(index: int) -> list[ModelResponse]:
    def call(name, **arguments):
        return ToolCall(id=name, name=name, arguments=arguments)

    return [
        ModelResponse(tool_calls=[call("shop_reset")]),
        ModelResponse(tool_calls=[call("shop_act", action=f"search[商品 {index}]")]),
        ModelResponse(tool_calls=[call("shop_act", action=f"click[A{index:04d}]")]),
        ModelResponse(tool_calls=[call("shop_act", action="click[黑色]")]),
        ModelResponse(tool_calls=[call("shop_act", action="click[buy now]")]),
    ]


def test_collection_counts_accepts_and_rejects_separately():
    pool = build_pool()
    backend = ScriptedBackend([*solve(0), ModelResponse(text="I give up")], repeat_last=True)
    runner = EpisodeRunner(backend, pool, max_turns=8)

    accepted, summary = collect_trajectories(tasks(2), runner, concurrency=1)

    assert summary.attempted == 2
    assert summary.accepted == 1
    assert summary.rejected == 1
    assert summary.reasons == {"turn_limit": 1}
    assert summary.mean_reward > 0
    assert len(accepted) == 1


def test_a_rejected_trajectory_is_still_written_down():
    pool = build_pool()
    backend = ScriptedBackend([ModelResponse(text="I give up")], repeat_last=True)
    runner = EpisodeRunner(backend, pool, max_turns=3)

    accepted, summary = collect_trajectories(tasks(1), runner, concurrency=1)

    assert accepted == []
    assert summary.rejected == 1


def test_the_writer_round_trips_a_trajectory_and_reports_what_exists(tmp_path):
    writer = TrajectoryWriter(tmp_path)
    path = writer.write(trajectory(7, termination="done", reward=0.5))

    assert path.exists() and writer.existing_ids() == {7}

