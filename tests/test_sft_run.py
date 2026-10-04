"""The SFT entry point, executed.

The trainer is the one component that nothing else can stand in for: a broken
optimiser argument, a collator that disagrees with the model, or a save path that
does not exist all look fine until a real run starts. This drives ``train()``
itself — the same function the GPU run calls — on a two-layer model built from
the real checkpoint's configuration and tokenizer.

It is a plumbing test, not a training run: two synthetic examples, one step, and
a sequence short enough to fit the container's 2 GB cap.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from shoprl.train.common import ModelConfig
from shoprl.train.sft import SftConfig, train

SEQUENCE = 64

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


def write_examples(path: Path, tokenizer) -> None:
    """Two short examples whose trained span stays at the end after trimming."""
    from shoprl.data.sft import build_turn_examples
    from shoprl.train.grpo import TOOLS

    record = {
        "task_id": 0,
        "metadata": {"session_id": "sft-run"},
        "contexts": [
            [
                {"role": "system", "content": "You are a shopping agent."},
                {"role": "user", "content": "buy a cup"},
            ]
        ],
        "targets": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "shop_reset", "arguments": {}},
                        }
                    ],
                }
            }
        ],
    }
    examples, rejected = build_turn_examples(record, tokenizer, tools=TOOLS, max_tokens=4096)
    assert examples and not rejected, rejected

    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            row = {
                "input_ids": example.input_ids[-SEQUENCE:],
                "labels": example.labels[-SEQUENCE:],
            }
            # The target is the tail of the sequence, so the trimmed row is still
            # trainable; anything else would be an empty gradient.
            assert any(label != -100 for label in row["labels"])
            handle.write(json.dumps(row) + "\n")


def tiny_model(tokenizer):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config(
        # len(), not .vocab_size: the chat template's special tokens sit above the
        # BPE vocabulary.
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        tie_word_embeddings=True,
    )
    return Qwen3ForCausalLM(config)


def test_the_sft_entry_point_trains_and_saves(tmp_path):
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    limit = memory_limit_gb()
    if 0 < limit < 1.5:
        pytest.skip(f"container memory limit is {limit:.1f} GB; too small for a training step")

    from transformers import AutoTokenizer

    checkpoint = require_checkpoint()

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    data = tmp_path / "turn_examples.jsonl"
    write_examples(data, tokenizer)

    output_dir = tmp_path / "run"
    config = SftConfig(
        model=ModelConfig(
            name_or_path=str(checkpoint),
            dtype="float32",
            gradient_checkpointing=False,
            lora_rank=0,
        ),
        data=data,
        output_dir=output_dir,
        epochs=1.0,
        batch_size=1,
        grad_accum=1,
        learning_rate=1e-3,
        max_seq_len=SEQUENCE,
        save_steps=1000,
        optimizer="adamw_torch",
    )

    stats = train(config, model=tiny_model(tokenizer), tokenizer=tokenizer)

    assert stats["examples"] == 1
    assert stats["total_steps"] == 1
    assert stats["trainable_parameters"] > 0
    assert transformers.__version__
    # A real checkpoint landed on disk, plus the manifest that makes it traceable.
    assert (output_dir / "config.json").exists()
    assert list(output_dir.glob("*.safetensors")) or (output_dir / "pytorch_model.bin").exists()
    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    assert manifest["examples"] == 1
    assert manifest["torch"]
