"""The whole CPU side of the pipeline, on the real catalogue and a real tokenizer.

teacher rollout -> trajectory JSON -> turn-level SFT examples, driven by a
scripted model that plays the task correctly. No API, no GPU, no reward model:
this proves the wiring that every expensive run depends on.

Skipped when the ShopSimulator release or a checkpoint is absent.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from shoprl.data.collect import TrajectoryWriter, collect_trajectories
from shoprl.env.catalog import load_catalog
from shoprl.env.local import EnvPool
from shoprl.env.search import Bm25Index
from shoprl.env.tasks import load_task_pool
from shoprl.harness.backends import ScriptedBackend
from shoprl.harness.rollout import EpisodeRunner
from shoprl.harness.types import ModelResponse, ToolCall


def require_checkpoint() -> Path:
    """Opt-in, like the other checkpoint-backed tests: see docs/devbox.md."""
    configured = os.environ.get("SHOPRL_TEST_CHECKPOINT")
    if not configured:
        pytest.skip("set SHOPRL_TEST_CHECKPOINT to run checkpoint-backed tests")
    path = Path(configured)
    if not (path / "config.json").exists():
        pytest.skip(f"no checkpoint at {path}")
    return path


def require_catalogue():
    try:
        catalogue = load_catalog()
    except FileNotFoundError:
        pytest.skip("ShopSimulator release not fetched")
    if len(catalogue) < 1000:
        pytest.skip("catalogue looks like the demo set")
    return catalogue


class OracleTeacher:
    """A scripted teacher that plays the page it was just shown, like a real one."""

    def __init__(self, catalogue, task_id: int):
        self.target = catalogue.at(task_id)
        self.phase = "reset"
        self.satisfied: set[int] = set()

    @staticmethod
    def _clickables(page: str) -> list[str]:
        marker = "可点击的按钮: "
        if marker not in page:
            return []
        try:
            return json.loads(page.split(marker, 1)[1].strip())
        except json.JSONDecodeError:
            return []

    def complete(self, messages, tools=None, **_kwargs) -> ModelResponse:
        page = next((m.content for m in reversed(messages) if m.role == "tool"), "")
        clickables = self._clickables(page)
        asin = self.target.asin.lower()

        if self.phase == "reset":
            self.phase = "search"
            return self._call("shop_reset")
        if self.phase in {"search", "paging"}:
            if asin in clickables:
                self.phase = "item"
                return self._call("shop_act", action=f"click[{asin}]")
            if self.phase == "search":
                self.phase = "paging"
                return self._call("shop_act", action=f"search[{self.target.instruction}]")
            if "next >" in clickables:
                return self._call("shop_act", action="click[next >]")
            return ModelResponse(text="not in these results")
        if self.phase == "item":
            navigation = {"buy now", "< prev", "back to search", "description", "features", "reviews"}
            options = [value for value in clickables if value not in navigation]
            for index, wanted in enumerate(self.target.instruction_options):
                if index in self.satisfied or not options:
                    continue
                from shoprl.env.reward import _ratio

                score, best = max(((_ratio(v, wanted.lower()), v) for v in options), default=(0.0, ""))
                if score > 85:
                    self.satisfied.add(index)
                    return self._call("shop_act", action=f"click[{best}]")
            if "buy now" in clickables:
                self.phase = "done"
                return self._call("shop_act", action="click[buy now]")
        return ModelResponse(text="no move left")

    @staticmethod
    def _call(name: str, **arguments) -> ModelResponse:
        return ModelResponse(tool_calls=[ToolCall(id=name, name=name, arguments=arguments)])


class OracleRunner(EpisodeRunner):
    """Rolls out with a fresh scripted teacher for each task."""

    def __init__(self, catalogue, pool, **kwargs):
        super().__init__(ScriptedBackend([], repeat_last=False), pool, **kwargs)
        self.catalogue = catalogue

    def run(self, task_id: int, **kwargs):
        self.backend = OracleTeacher(self.catalogue, task_id)
        return super().run(task_id, **kwargs)


def test_teacher_rollouts_become_trainable_turn_examples(tmp_path):
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    # Check the cheap preconditions before loading a 23k-product catalogue and
    # building an index over it, so a skip costs nothing.
    checkpoint = require_checkpoint()
    catalogue = require_catalogue()

    pool = EnvPool(catalogue, Bm25Index.build(catalogue), capacity=2)
    tasks = [load_task_pool("dev")[0]]
    task_id = tasks[0].task_id
    runner = OracleRunner(catalogue, pool, max_turns=20)

    accepted, summary = collect_trajectories(tasks, runner, concurrency=1)

    assert summary.attempted == 1
    if not accepted:
        pytest.skip(f"oracle did not finish task {task_id}: {summary.reasons}")

    writer = TrajectoryWriter(tmp_path / "raw")
    writer.write(accepted[0])
    written = list((tmp_path / "raw").glob("*.json"))
    assert written, "trajectory was not persisted"
    assert writer.existing_ids() == {task_id}

    from shoprl.data.sft import write_sft_dataset
    from shoprl.train.grpo import TOOLS

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    records = [json.loads(path.read_text(encoding="utf-8")) for path in written]
    result = write_sft_dataset(
        records, tokenizer, tmp_path / "prepared", tools=TOOLS, max_tokens=16384
    )

    assert result.accepted == 1
    assert result.examples > 0, result.to_json()
    assert result.target_mean > 0
    assert (tmp_path / "prepared" / "turn_examples.jsonl").exists()
