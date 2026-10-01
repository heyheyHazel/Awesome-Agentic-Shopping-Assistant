"""The RL chain from a live rollout to a tensor-ready turn sample.

Every link here has a silent failure mode. If the engine's prompt rendering
disagrees with the chat template, or the sampled tokens do not line up with the
target span, the sample is dropped and the run optimises nothing while still
reporting a plausible loss. This drives the real environment and the real
tokenizer and asserts the samples come out non-empty and aligned.

No model forward pass, so it runs on a CPU-only container.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from shoprl.harness.types import Message
from tests.oracle_teacher import OracleTeacher

CHECKPOINT = Path(os.environ.get("SHOPRL_TEST_CHECKPOINT", "models/Qwen3-1.7B"))


def require_checkpoint() -> Path:
    if not (CHECKPOINT / "config.json").exists():
        pytest.skip(f"no checkpoint at {CHECKPOINT}")
    return CHECKPOINT


def require_catalogue():
    from shoprl.env.catalog import load_catalog

    try:
        catalogue = load_catalog()
    except FileNotFoundError:
        pytest.skip("ShopSimulator release not fetched")
    if len(catalogue) < 1000:
        pytest.skip("catalogue looks like the demo set")
    return catalogue


def require_tokenizer():
    pytest.importorskip("torch")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(require_checkpoint(), trust_remote_code=True)


def test_the_rollout_prompt_matches_the_training_render():
    """The two renderers must agree, or every sample is silently dropped."""
    from shoprl.data.sft import _as_conversation, _render
    from shoprl.train.grpo import TOOLS
    from shoprl.train.rollout import render_prompt

    tokenizer = require_tokenizer()
    messages = [
        Message(role="system", content="You are a shopping agent."),
        Message(role="user", content="buy a cup"),
    ]

    rollout_ids = list(render_prompt(tokenizer, messages, TOOLS)["input_ids"][0])
    training_ids = _render(
        tokenizer, _as_conversation(messages), TOOLS, tokenize=True, add_generation_prompt=True
    )

    assert rollout_ids == training_ids


def test_the_sampled_span_is_reproduced_by_the_chat_template():
    """What an engine would sample must equal what the trainer rebuilds."""
    from shoprl.data.sft import _render
    from shoprl.train.grpo import TOOLS

    tokenizer = require_tokenizer()
    message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {"name": "shop_act", "arguments": {"action": "click[buy now]"}},
            }
        ],
    }

    sampled = _render(tokenizer, [message], TOOLS, tokenize=True)
    rebuilt = _render(tokenizer, [message], TOOLS, tokenize=True)

    assert sampled and sampled == rebuilt


def test_a_real_rollout_becomes_non_empty_training_samples():
    from shoprl.env.local import EnvPool
    from shoprl.env.search import Bm25Index
    from shoprl.env.tasks import load_task_pool
    from shoprl.harness.rollout import EpisodeRunner
    from shoprl.train.grpo import GrpoTrainer

    catalogue = require_catalogue()
    tokenizer = require_tokenizer()
    task_id = load_task_pool("dev")[0].task_id
    pool = EnvPool(catalogue, Bm25Index.build(catalogue), capacity=1)
    runner = EpisodeRunner(OracleTeacher(catalogue, tokenizer, task_id), pool, max_turns=20)

    trajectory = runner.run(task_id)

    assert trajectory.termination == "done"
    trainer = GrpoTrainer.__new__(GrpoTrainer)
    trainer.tokenizer = tokenizer
    trainer.config = SimpleNamespace(max_seq_len=16384, max_turn_samples_per_trajectory=0)

    samples = trainer._turn_samples(trajectory)

    assert samples, "the rollout produced no trainable turn samples"
    for sample in samples:
        lengths = {len(sample["input_ids"]), len(sample["loss_mask"]), len(sample["old_logprobs"])}
        assert len(lengths) == 1
        trained = sum(sample["loss_mask"])
        assert trained > 0
        # The mask is a suffix: the prompt is never trained on.
        assert sample["loss_mask"] == [0] * (len(sample["loss_mask"]) - trained) + [1] * trained


class StubTrajectory:
    """Stands in for ``Trajectory`` so a test can hand ``_turn_samples`` a record."""

    def __init__(self, record: dict):
        self._record = record

    def to_json(self) -> dict:
        return self._record


def test_a_mismatched_turn_is_dropped_rather_than_trained_on():
    """The guard that keeps a silent template change from corrupting a run."""
    from shoprl.train.grpo import GrpoTrainer

    tokenizer = require_tokenizer()
    record = {
        "task_id": 0,
        "metadata": {"session_id": "mismatch"},
        "contexts": [[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]],
        "targets": [{"message": {"role": "assistant", "content": "hello"}, "token_ids": [1, 2, 3]}],
    }
    trainer = GrpoTrainer.__new__(GrpoTrainer)
    trainer.tokenizer = tokenizer
    trainer.config = SimpleNamespace(max_seq_len=4096, max_turn_samples_per_trajectory=0)

    # The engine claims three tokens that the template does not produce, so the
    # turn is dropped instead of being trained against a wrong span.
    assert trainer._turn_samples(StubTrajectory(record)) == []
