"""The GRPO loop, executed.

The RL entry point is the piece with the most moving parts and the least
evidence: it loads tasks, rolls out a group of candidates, centres their rewards,
converts the trajectories, updates, logs and checkpoints. This runs all of it —
the function the GPU run calls — with a scripted engine standing in for the
policy's sampling.

Two candidates with different outcomes are deliberate: a group with no spread
produces zero advantages and skips the update entirely, which would leave the
interesting half of the loop unexecuted.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from shoprl.data.scripted import ReluctantTeacher
from shoprl.train.common import ModelConfig
from shoprl.train.grpo import GrpoConfig, GrpoTrainer

pytestmark = pytest.mark.slow


def require_checkpoint() -> Path:
    configured = os.environ.get("SHOPRL_TEST_CHECKPOINT")
    if not configured:
        pytest.skip("set SHOPRL_TEST_CHECKPOINT to run checkpoint-backed tests")
    path = Path(configured)
    if not (path / "config.json").exists():
        pytest.skip(f"no checkpoint at {path}")
    return path


def memory_limit_gb() -> float:
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            value = Path(path).read_text().strip()
        except OSError:
            continue
        if value.isdigit():
            return int(value) / 2**30
    return 0.0


def tiny_model(tokenizer):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    return Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=len(tokenizer),
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=512,
            tie_word_embeddings=True,
        )
    )


def build_trainer(tmp_path, *, max_seq_len: int):
    """A GRPO trainer wired to the real catalogue, a scripted engine and a tiny model."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    checkpoint = require_checkpoint()

    from shoprl.env.catalog import load_catalog
    from shoprl.env.local import EnvPool
    from shoprl.env.search import Bm25Index
    from shoprl.env.tasks import load_task_pool

    catalogue = load_catalog()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    task_id = load_task_pool("dev")[0].task_id
    pool = EnvPool(catalogue, Bm25Index.build(catalogue), capacity=1)

    config = GrpoConfig(
        model=ModelConfig(
            name_or_path=str(checkpoint),
            dtype="float32",
            gradient_checkpointing=False,
            lora_rank=0,
        ),
        output_dir=tmp_path / "grpo",
        tasks_name="dev",
        steps=1,
        group_size=2,
        tasks_per_step=1,
        max_turns=8,
        max_tool_calls=10,
        max_response_tokens=32,
        max_context_tokens=4096,
        max_seq_len=max_seq_len,
        learning_rate=1e-3,
        seed=7,
        save_every=1,
        max_turn_samples_per_trajectory=1,
    )
    trainer = GrpoTrainer(
        config,
        engine=ReluctantTeacher(catalogue, task_id, tokenizer),
        pool=pool,
        model=tiny_model(tokenizer),
        tokenizer=tokenizer,
    )
    return trainer, config


def test_the_grpo_loop_rolls_out_scores_and_checkpoints(tmp_path):
    """Everything except the update's backward pass, which needs more memory."""
    # No turn survives the length cap, so the update has nothing to do and the
    # loop still has to load tasks, roll out a group, score it, log and save.
    trainer, config = build_trainer(tmp_path, max_seq_len=256)

    history = trainer.train()

    assert len(history) == 1
    step = history[0]
    assert step["rollouts"] == 2
    # One candidate buys, the other gives up: the group carries signal.
    assert step["zero_variance_groups"] == 0
    assert step["mean_reward"] > 0

    output_dir = config.output_dir
    logged = json.loads((output_dir / "train_log.jsonl").read_text().strip().splitlines()[0])
    assert logged["step"] == 0

    checkpoint = output_dir / f"step-{config.steps}"
    assert (checkpoint / "run_manifest.json").exists()
    manifest = json.loads((checkpoint / "run_manifest.json").read_text())
    assert manifest["group_size"] == 2
    assert manifest["base_model"] == str(checkpoint)
    assert SimpleNamespace(**manifest).seed == config.seed


def test_the_grpo_update_takes_a_gradient_step(tmp_path):
    """The same loop with the samples kept, so the policy loss and backward run.

    A sampled turn is a couple of thousand tokens and the vocabulary is 151k, so
    the backward pass needs several GB. It runs in CI and on a host with a card
    allocated; here it skips rather than being killed mid-run.
    """
    limit = memory_limit_gb()
    if 0 < limit < 6:
        pytest.skip(f"container memory limit is {limit:.1f} GB; the backward pass needs ~6 GB")

    trainer, _ = build_trainer(tmp_path, max_seq_len=4096)
    step = trainer.train()[0]

    assert step["turn_samples"] > 0
    assert step["loss"] > 0
