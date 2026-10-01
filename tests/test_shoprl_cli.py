"""CLI wiring: the config file has to reach the objects the trainer builds.

Every flag used to carry a non-None default, which is indistinguishable from a
flag the user typed, so ``--model`` in a config file was silently replaced by the
built-in default. That is invisible until a run trains the wrong checkpoint.
"""

from __future__ import annotations

import json

import pytest

from shoprl.cli import main


def write_config(tmp_path, **values) -> str:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(values), encoding="utf-8")
    return str(path)


def test_the_config_file_model_reaches_the_trainer(monkeypatch, tmp_path):
    captured = {}

    def fake_train(config):
        captured["config"] = config
        return {"ok": True}

    monkeypatch.setattr("shoprl.train.sft.train", fake_train)
    config_path = write_config(
        tmp_path,
        model="models/Qwen3-0.6B",
        data="runs/sft_data/turn_examples.jsonl",
        output_dir="runs/tiny-sft",
        lora_rank=16,
        dtype="float16",
        learning_rate=3e-5,
    )

    assert main(["sft", "--config", config_path]) == 0

    config = captured["config"]
    assert config.model.name_or_path == "models/Qwen3-0.6B"
    assert config.model.lora_rank == 16
    assert config.model.dtype == "float16"
    assert config.learning_rate == pytest.approx(3e-5)
    assert str(config.output_dir) == "runs/tiny-sft"


def test_a_flag_still_overrides_the_config_file(monkeypatch, tmp_path):
    captured = {}

    def fake_train(config):
        captured["config"] = config
        return {"ok": True}

    monkeypatch.setattr("shoprl.train.sft.train", fake_train)
    config_path = write_config(
        tmp_path, model="models/Qwen3-0.6B", data="d.jsonl", output_dir="out"
    )

    main(["sft", "--config", config_path, "--model", "models/Qwen3-4B", "--lora-rank", "8"])

    assert captured["config"].model.name_or_path == "models/Qwen3-4B"
    assert captured["config"].model.lora_rank == 8


def test_a_config_missing_a_required_field_says_so(tmp_path):
    config_path = write_config(tmp_path, model="models/Qwen3-0.6B", output_dir="out")

    with pytest.raises(SystemExit) as error:
        main(["sft", "--config", config_path])

    assert "data is required" in str(error.value)


def test_the_shipped_configs_parse_and_exist():
    from pathlib import Path

    for name in ("sft.json", "grpo.json", "grpo_rlvr_opd.json"):
        path = Path("configs") / name
        assert path.exists(), name
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["model"], f"{name} has no model"
        # A hub id would need network the training box does not have for
        # huggingface.co, so the shipped configs point at the downloaded copies.
        assert payload["model"].startswith("models/"), f"{name}: {payload['model']}"

