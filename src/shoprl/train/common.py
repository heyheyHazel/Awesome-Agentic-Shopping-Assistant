"""Shared training plumbing: model loading, precision, LoRA, reproducibility."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ModelConfig:
    """Checkpoint, precision and adapter choices shared by SFT and RL."""

    name_or_path: str = "Qwen/Qwen3-1.7B"
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    gradient_checkpointing: bool = True
    lora_rank: int = 0
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_targets: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        import torch

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:  # a CPU-only host can still prepare data
        pass


def resolve_dtype(name: str):
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def precision_kwarg(config: ModelConfig) -> dict[str, Any]:
    """Keyword ``from_pretrained`` wants for the weight dtype.

    transformers 5 renamed ``torch_dtype`` to ``dtype``. Both spellings are
    accepted at different versions, and passing the wrong one is a hard failure
    before any weights load, so the choice is made once here.
    """
    import transformers

    major = int(transformers.__version__.split(".")[0])
    name = "dtype" if major >= 5 else "torch_dtype"
    return {name: resolve_dtype(config.dtype)}


def load_tokenizer(config: ModelConfig):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(config: ModelConfig, *, trainable: bool = True, adapter: str | None = None):
    """Load the policy, optionally wrapping it in a fresh or existing LoRA adapter."""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        config.name_or_path,
        attn_implementation=config.attn_implementation,
        trust_remote_code=True,
        **precision_kwarg(config),
    )
    if config.lora_rank > 0:
        from peft import LoraConfig, PeftModel, get_peft_model

        if adapter:
            model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
        else:
            model = get_peft_model(
                model,
                LoraConfig(
                    r=config.lora_rank,
                    lora_alpha=config.lora_alpha,
                    lora_dropout=config.lora_dropout,
                    target_modules=config.lora_targets,
                    task_type="CAUSAL_LM",
                ),
            )
        if trainable:
            model.enable_input_require_grads()
    elif not trainable:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    if config.gradient_checkpointing and trainable:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
    return model


def write_run_manifest(run_dir: Path, payload: dict[str, Any]) -> None:
    """Record what produced a checkpoint: seeds, config and environment versions."""
    import platform

    import torch

    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        **payload,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
