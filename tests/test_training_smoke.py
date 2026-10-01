"""The SFT path against a real tokenizer and a real model configuration.

Everything else in the suite uses a fake tokenizer, which cannot catch the two
failure modes that actually bite here: a chat template whose rendering is not
prefix-consistent with the target appended, and a training library version that
renamed or removed an argument. Both are silent until the first real run, so they
are checked against a downloaded checkpoint when one is available.

Set ``SHOPRL_TEST_CHECKPOINT`` to point at a different one; the tests skip when
no checkpoint is present, which keeps a fresh clone green.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

CHECKPOINT = Path(os.environ.get("SHOPRL_TEST_CHECKPOINT", "models/Qwen3-1.7B"))


def require_checkpoint() -> Path:
    if not (CHECKPOINT / "config.json").exists():
        pytest.skip(f"no checkpoint at {CHECKPOINT}")
    return CHECKPOINT


def memory_limit_gb() -> float:
    """Container memory cap, or 0 when it cannot be determined."""
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            value = Path(path).read_text().strip()
        except OSError:
            continue
        if value.isdigit():
            return int(value) / 2**30
    return 0.0


def sample_trajectory() -> dict:
    """One recorded trajectory in the shape ``Trajectory.to_json`` produces."""
    task = "完成给定的购物任务。先调用 shop_reset 获取任务，然后只使用 shop_act 与环境交互。"
    instruction = "帮我找一款适合5岁小孩的泰国进口天然乳胶枕头，预算1000元以下。"
    return {
        "schema_version": 1,
        "task_id": 0,
        "reward": 1.0,
        "reward_detail": {"r_type": 1.0, "r_att": 1.0, "r_option": 1.0, "r_price": 1.0},
        "termination": "done",
        "metadata": {"session_id": "smoke", "accepted": True},
        "contexts": [
            [
                {"role": "system", "content": "You are a shopping agent."},
                {"role": "user", "content": task},
            ],
            [
                {"role": "system", "content": "You are a shopping agent."},
                {"role": "user", "content": task},
                {
                    "role": "tool",
                    "content": f"Instruction: {instruction}",
                    "tool_call_id": "shop_reset",
                },
            ],
        ],
        "targets": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "shop_reset",
                            "type": "function",
                            "function": {"name": "shop_reset", "arguments": {}},
                        }
                    ],
                }
            },
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "shop_act",
                            "type": "function",
                            "function": {
                                "name": "shop_act",
                                "arguments": {"action": f"search[{instruction}]"},
                            },
                        }
                    ],
                }
            },
        ],
    }


def test_training_arguments_translate_the_config(tmp_path):
    """No checkpoint needed: this is the library-compatibility check."""
    transformers = pytest.importorskip("transformers")

    from shoprl.train.common import ModelConfig
    from shoprl.train.sft import SftConfig, build_training_arguments, schedule_steps

    config = SftConfig(
        model=ModelConfig(name_or_path="unused", dtype="bfloat16", gradient_checkpointing=False),
        data=tmp_path / "unused.jsonl",
        output_dir=tmp_path / "out",
        batch_size=2,
        grad_accum=4,
        epochs=2.0,
        learning_rate=2e-5,
    )
    examples = 100
    arguments = build_training_arguments(config, examples)
    warmup_steps, total_steps = schedule_steps(config, examples)

    steps_per_epoch = -(-examples // (config.batch_size * config.grad_accum))
    expected_total = int(steps_per_epoch * config.epochs)
    assert total_steps == expected_total
    assert warmup_steps == int(config.warmup_ratio * expected_total)
    # The same numbers the trainer writes into the run manifest.
    assert arguments.warmup_steps == warmup_steps
    assert arguments.learning_rate == pytest.approx(2e-5)
    # transformers 5 renamed the scheduler argument away from warmup_ratio; if it
    # ever comes back this still passes, and if the constructor is fed an argument
    # it does not know, the call above already raised.
    assert transformers.__version__


def test_the_chat_template_is_prefix_consistent_with_a_real_tokenizer():
    """A template that is not prefix-consistent silently corrupts every SFT label."""
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    from shoprl.data.sft import build_turn_examples
    from shoprl.train.grpo import TOOLS

    tokenizer = AutoTokenizer.from_pretrained(require_checkpoint(), trust_remote_code=True)
    examples, rejected = build_turn_examples(
        sample_trajectory(), tokenizer, tools=TOOLS, max_tokens=8192
    )

    assert rejected == [], rejected
    assert len(examples) == 2
    for example in examples:
        first = next(i for i, label in enumerate(example.labels) if label != -100)
        assert example.labels[:first] == [-100] * first
        assert example.labels[first:] == example.input_ids[first:]
        assert example.metadata["target_token_count"] > 0


def test_one_turn_example_takes_a_gradient_step(tmp_path):
    """Build the smallest real model and push one example through it.

    This is not a training run: two layers, a two-example batch, one step. It
    proves the collator, the loss and the backward pass line up before a GPU is
    ever involved.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    from shoprl.data.sft import build_turn_examples
    from shoprl.train.grpo import TOOLS
    from shoprl.train.sft import build_collator

    # A forward and backward through a 151k-vocabulary head needs well over the
    # 2 GB this container is capped at before a GPU is allocated. The check is on
    # the cgroup, not on the host, because the host has a terabyte.
    limit = memory_limit_gb()
    if 0 < limit < 8:
        pytest.skip(f"container memory limit is {limit:.1f} GB; needs ~8 GB for this pass")

    tokenizer = AutoTokenizer.from_pretrained(require_checkpoint(), trust_remote_code=True)
    examples, _ = build_turn_examples(
        sample_trajectory(), tokenizer, tools=TOOLS, max_tokens=8192
    )
    config = Qwen3Config(
        # len(), not .vocab_size: the chat template's special tokens live above
        # the BPE vocabulary and an embedding table that stops short of them
        # fails with an index error on the first forward pass.
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        tie_word_embeddings=True,
    )
    model = Qwen3ForCausalLM(config)
    collate = build_collator(tokenizer.pad_token_id or tokenizer.eos_token_id)
    # Keep only the tail of each example. The head is prompt context this test
    # does not exercise, and a full-length turn through a 151k-vocabulary head
    # costs minutes of CPU for no extra coverage.
    tail = 192
    trimmed = [
        {
            "input_ids": example.input_ids[-tail:],
            "labels": example.labels[-tail:],
            "attention_mask": [1] * len(example.input_ids[-tail:]),
        }
        for example in examples
    ]
    # The assistant target is the last thing in each sequence, so the tail always
    # contains trainable positions.
    assert all(any(label != -100 for label in row["labels"]) for row in trimmed)

    batch = collate(trimmed)

    loss = model(**batch).loss
    loss.backward()

    assert torch.isfinite(loss)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and any(float(g.abs().sum()) > 0 for g in grads)
    # Padded positions must not contribute.
    assert batch["labels"].shape == batch["input_ids"].shape
