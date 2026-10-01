"""Supervised fine-tuning on turn-level shopping trajectories.

Labels are precomputed by ``shoprl.data.sft``, so the trainer never has to guess
which tokens are trainable. That matters here more than in ordinary SFT: most of
each sequence is tool output, and a collator that masked by role rather than by
token would train the model to write the pages it is supposed to read.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shoprl.train.common import (
    ModelConfig,
    load_model,
    load_tokenizer,
    set_seed,
    write_run_manifest,
)

IGNORE_INDEX = -100


@dataclass(slots=True)
class SftConfig:
    model: ModelConfig
    data: Path
    output_dir: Path
    epochs: float = 1.0
    batch_size: int = 1
    grad_accum: int = 8
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    max_seq_len: int = 16384
    optimizer: str = "adamw_8bit"
    save_steps: int = 200
    logging_steps: int = 1
    seed: int = 1234
    report_to: str = "none"
    resume_from: str = ""


class TurnDataset:
    """Reads ``turn_examples.jsonl``; longest examples are dropped, never truncated."""

    def __init__(self, path: Path, max_seq_len: int):
        self.rows: list[dict[str, Any]] = []
        self.skipped = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if len(row["input_ids"]) > max_seq_len:
                    self.skipped += 1
                    continue
                self.rows.append(row)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        return {
            "input_ids": row["input_ids"],
            "labels": row["labels"],
            "attention_mask": [1] * len(row["input_ids"]),
        }


def build_collator(pad_token_id: int):
    import torch

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        width = max(len(row["input_ids"]) for row in batch)

        def pad(values: list[int], fill: int) -> torch.Tensor:
            return torch.tensor(
                [values + [fill] * (width - len(values)) for values in batch], dtype=torch.long
            )

        return {
            "input_ids": pad([row["input_ids"] for row in batch], pad_token_id),
            "labels": pad([row["labels"] for row in batch], IGNORE_INDEX),
            "attention_mask": pad([row["attention_mask"] for row in batch], 0),
        }

    return collate


def train(config: SftConfig) -> dict[str, Any]:
    import torch
    from transformers import Trainer, TrainingArguments

    set_seed(config.seed)
    tokenizer = load_tokenizer(config.model)
    model = load_model(config.model)
    dataset = TurnDataset(config.data, config.max_seq_len)
    if not len(dataset):
        raise SystemExit(f"no trainable examples under {config.max_seq_len} tokens in {config.data}")

    # Only trainable parameters are handed to the optimizer; a frozen base under a
    # LoRA adapter must not be given weight decay or its own optimizer state.
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    uses_lora = config.model.lora_rank > 0

    arguments = TrainingArguments(
        output_dir=str(config.output_dir),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=2,
        bf16=config.model.dtype == "bfloat16",
        fp16=config.model.dtype == "float16",
        optim=config.optimizer,
        gradient_checkpointing=config.model.gradient_checkpointing,
        report_to=[] if config.report_to == "none" else [config.report_to],
        seed=config.seed,
        data_seed=config.seed,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=dataset,
        data_collator=build_collator(tokenizer.pad_token_id),
        optimizers=(
            torch.optim.AdamW(trainable, lr=config.learning_rate),
            None,
        )
        if uses_lora
        else (None, None),
    )
    trainer.train(resume_from_checkpoint=config.resume_from or None)
    trainer.save_model(str(config.output_dir))
    tokenizer.save_pretrained(str(config.output_dir))

    stats = {
        "examples": len(dataset),
        "skipped_over_length": dataset.skipped,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "grad_accum": config.grad_accum,
        "learning_rate": config.learning_rate,
        "max_seq_len": config.max_seq_len,
        "seed": config.seed,
        "data": str(config.data),
        "model": config.model.name_or_path,
        "lora_rank": config.model.lora_rank,
    }
    write_run_manifest(config.output_dir, stats)
    return stats

