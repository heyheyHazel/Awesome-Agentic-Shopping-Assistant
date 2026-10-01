"""One entry point for the whole pipeline.

Every stage is a subcommand so a run is reproducible from a shell history:

    python -m shoprl.cli catalogue                     # raw release -> product cache
    python -m shoprl.cli tasks                         # task pools and splits
    python -m shoprl.cli collect --tasks sft ...       # teacher trajectories
    python -m shoprl.cli prepare-sft --input ...       # turn-level SFT data
    python -m shoprl.cli sft --config configs/sft.json
    python -m shoprl.cli grpo --config configs/grpo.json
    python -m shoprl.cli eval --tasks official_test --limit 200

Config files are JSON, then any field can be overridden on the command line
(``--learning_rate 1e-5``). Every run writes its resolved config into
``run_manifest.json`` next to the artefacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _json_config(path: str | None, overrides: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8")) if path else {}
    payload.update({key: value for key, value in overrides.items() if value is not None})
    return payload


def _overrides(args: argparse.Namespace, keys: list[str]) -> dict[str, Any]:
    return {key: getattr(args, key, None) for key in keys}


def _add_config_flags(parser: argparse.ArgumentParser, keys: dict[str, tuple[type, str]]) -> None:
    for name, (kind, help_text) in keys.items():
        parser.add_argument(f"--{name}", type=kind, default=None, help=help_text)


# ── subcommands ───────────────────────────────────────────────────────


def cmd_catalogue(args: argparse.Namespace) -> int:
    from shoprl.env.catalog import build_catalog, load_catalog

    if args.force:
        print(json.dumps(build_catalog(), ensure_ascii=False, indent=2))
    catalog = load_catalog(rebuild=args.force)
    print(json.dumps(catalog.manifest(), ensure_ascii=False, indent=2))
    return 0


def cmd_tasks(args: argparse.Namespace) -> int:
    from shoprl.env.tasks import build_task_pools

    sizes = {"dev": args.dev, "sft": args.sft, "rl": args.rl}
    print(json.dumps(build_task_pools(sizes=sizes, seed=args.seed), ensure_ascii=False, indent=2))
    return 0


def cmd_collect(args: argparse.Namespace) -> int:
    from shoprl.data.collect import TrajectoryWriter, collect_trajectories
    from shoprl.env.catalog import load_catalog
    from shoprl.env.local import EnvPool
    from shoprl.env.search import Bm25Index
    from shoprl.env.tasks import load_task_pool
    from shoprl.harness.backends import OpenAIBackend
    from shoprl.harness.context import ContextPolicy
    from shoprl.harness.rollout import EpisodeRunner

    catalog = load_catalog()
    pool = EnvPool(catalog, Bm25Index.build(catalog), capacity=args.concurrency)
    backend = OpenAIBackend(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key or "EMPTY",
        timeout=args.timeout,
    )
    runner = EpisodeRunner(
        backend,
        pool,
        context_policy=ContextPolicy(keep_recent_tool_results=args.keep_tool_results),
        max_turns=args.max_turns,
        max_tool_calls=args.max_tool_calls,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        show_persona=args.persona,
    )
    tasks = load_task_pool(args.tasks)
    if args.limit:
        tasks = tasks[: args.limit]

    out = Path(args.out)
    writer = TrajectoryWriter(out / "raw")
    accepted, summary = collect_trajectories(
        tasks,
        runner,
        samples_per_task=args.samples_per_task,
        concurrency=args.concurrency,
        min_reward=args.min_reward,
        on_trajectory=writer.write,
    )
    (out / "summary.json").write_text(
        json.dumps(summary.to_json(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary.to_json(), ensure_ascii=False, indent=2))
    print(f"accepted {len(accepted)} trajectories -> {out}")
    return 0


def cmd_prepare_sft(args: argparse.Namespace) -> int:
    from shoprl.data.collect import load_trajectories
    from shoprl.data.sft import write_sft_dataset
    from shoprl.train.common import load_tokenizer
    from shoprl.train.grpo import TOOLS

    records = load_trajectories(Path(args.input) / "raw")
    tokenizer = load_tokenizer(_model_config(args))
    summary = write_sft_dataset(
        records,
        tokenizer,
        Path(args.out),
        tools=TOOLS,
        max_tokens=args.max_tokens,
        verify_template=not args.skip_template_check,
    )
    print(json.dumps(summary.to_json(), ensure_ascii=False, indent=2))
    return 0


def _model_config(args: argparse.Namespace):
    from shoprl.train.common import ModelConfig

    return ModelConfig(
        name_or_path=args.model,
        dtype=args.dtype,
        attn_implementation=args.attn,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )


def cmd_sft(args: argparse.Namespace) -> int:
    from shoprl.train.sft import SftConfig, train

    values = _json_config(
        args.config,
        _overrides(
            args,
            [
                "model", "data", "output_dir", "epochs", "batch_size", "grad_accum",
                "learning_rate", "max_seq_len", "lora_rank", "dtype", "save_steps", "seed",
            ],
        ),
    )
    config = SftConfig(
        model=_model_config(args),
        data=Path(values["data"]),
        output_dir=Path(values["output_dir"]),
        epochs=float(values.get("epochs", 1.0)),
        batch_size=int(values.get("batch_size", 1)),
        grad_accum=int(values.get("grad_accum", 8)),
        learning_rate=float(values.get("learning_rate", 1e-5)),
        max_seq_len=int(values.get("max_seq_len", 16384)),
        save_steps=int(values.get("save_steps", 200)),
        seed=int(values.get("seed", 1234)),
    )
    print(json.dumps(train(config), ensure_ascii=False, indent=2))
    return 0


def cmd_grpo(args: argparse.Namespace) -> int:
    from shoprl.train.distill import DistillConfig
    from shoprl.train.grpo import GrpoConfig, GrpoTrainer

    values = _json_config(
        args.config,
        _overrides(
            args,
            [
                "output_dir", "tasks_name", "steps", "group_size", "tasks_per_step",
                "learning_rate", "max_response_tokens", "max_context_tokens", "kl_coef",
                "temperature", "seed", "concurrency", "init_adapter", "lora_rank", "dtype",
            ],
        ),
    )
    distill = None
    if args.distill_beta > 0:
        distill = DistillConfig(
            mode=args.distill_mode,
            beta=args.distill_beta,
            action_tokens_only=args.distill_action_tokens_only,
        )
    config = GrpoConfig(
        model=_model_config(args),
        output_dir=Path(values["output_dir"]),
        tasks_name=values.get("tasks_name", "rl"),
        steps=int(values.get("steps", 100)),
        group_size=int(values.get("group_size", 4)),
        tasks_per_step=int(values.get("tasks_per_step", 5)),
        learning_rate=float(values.get("learning_rate", 1e-6)),
        max_response_tokens=int(values.get("max_response_tokens", 256)),
        max_context_tokens=int(values.get("max_context_tokens", 16384)),
        kl_coef=float(values.get("kl_coef", 0.0)),
        temperature=float(values.get("temperature", 1.0)),
        seed=int(values.get("seed", 1234)),
        concurrency=int(values.get("concurrency", 4)),
        init_adapter=values.get("init_adapter", ""),
        distill=distill,
        teacher_path=args.teacher_path,
        self_teacher=args.self_teacher,
    )
    print(json.dumps(GrpoTrainer(config).train(), ensure_ascii=False, indent=2))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from shoprl.env.catalog import load_catalog
    from shoprl.env.local import EnvPool
    from shoprl.env.search import Bm25Index
    from shoprl.eval.run import EvalConfig, evaluate
    from shoprl.harness.backends import OpenAIBackend
    from shoprl.harness.context import ContextPolicy
    from shoprl.harness.rollout import EpisodeRunner

    catalog = load_catalog()
    pool = EnvPool(catalog, Bm25Index.build(catalog), capacity=args.concurrency)
    backend = OpenAIBackend(
        model=args.model, base_url=args.base_url, api_key=args.api_key or "EMPTY"
    )
    runner = EpisodeRunner(
        backend,
        pool,
        context_policy=ContextPolicy(keep_recent_tool_results=args.keep_tool_results),
        max_turns=args.max_turns,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        show_persona=args.persona,
    )
    metrics = evaluate(
        runner,
        EvalConfig(
            output_dir=Path(args.out),
            tasks_name=args.tasks,
            samples_per_task=args.samples,
            limit=args.limit,
            concurrency=args.concurrency,
            temperature=args.temperature,
            label=args.label or args.model,
        ),
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


# ── parser ────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="shoprl", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    catalogue = sub.add_parser("catalogue", help="build the product cache from the raw release")
    catalogue.add_argument("--force", action="store_true")
    catalogue.set_defaults(func=cmd_catalogue)

    tasks = sub.add_parser("tasks", help="build the task pools")
    tasks.add_argument("--dev", type=int, default=150)
    tasks.add_argument("--sft", type=int, default=512)
    tasks.add_argument("--rl", type=int, default=500)
    tasks.add_argument("--seed", type=int, default=20260101)
    tasks.set_defaults(func=cmd_tasks)

    collect = sub.add_parser("collect", help="collect teacher trajectories")
    collect.add_argument("--tasks", default="sft")
    collect.add_argument("--out", required=True)
    collect.add_argument("--model", required=True)
    collect.add_argument("--base-url", default="https://api.deepseek.com/v1")
    collect.add_argument("--api-key", default="")
    collect.add_argument("--limit", type=int, default=0)
    collect.add_argument("--samples-per-task", type=int, default=1)
    collect.add_argument("--concurrency", type=int, default=4)
    collect.add_argument("--temperature", type=float, default=0.7)
    collect.add_argument("--max-tokens", type=int, default=1024)
    collect.add_argument("--max-turns", type=int, default=30)
    collect.add_argument("--max-tool-calls", type=int, default=60)
    collect.add_argument("--keep-tool-results", type=int, default=3)
    collect.add_argument("--min-reward", type=float, default=0.0)
    collect.add_argument("--timeout", type=float, default=300.0)
    collect.add_argument("--persona", action="store_true", help="show the shopper's persona on reset")
    collect.set_defaults(func=cmd_collect)

    prepare = sub.add_parser("prepare-sft", help="turn trajectories into SFT examples")
    prepare.add_argument("--input", required=True)
    prepare.add_argument("--out", required=True)
    prepare.add_argument("--model", default="Qwen/Qwen3-1.7B")
    prepare.add_argument("--max-tokens", type=int, default=16384)
    prepare.add_argument("--skip-template-check", action="store_true")
    _add_model_flags(prepare)
    prepare.set_defaults(func=cmd_prepare_sft)

    sft = sub.add_parser("sft", help="supervised fine-tuning")
    sft.add_argument("--config")
    sft.add_argument("--model", default="Qwen/Qwen3-1.7B")
    sft.add_argument("--data", required=True)
    sft.add_argument("--output-dir", required=True)
    sft.add_argument("--epochs", type=float)
    sft.add_argument("--batch-size", type=int)
    sft.add_argument("--grad-accum", type=int)
    sft.add_argument("--learning-rate", type=float)
    sft.add_argument("--max-seq-len", type=int)
    sft.add_argument("--save-steps", type=int)
    sft.add_argument("--seed", type=int)
    _add_model_flags(sft)
    sft.set_defaults(func=cmd_sft)

    grpo = sub.add_parser("grpo", help="GRPO with verifiable rewards (+ optional distillation)")
    grpo.add_argument("--config")
    grpo.add_argument("--model", default="Qwen/Qwen3-1.7B")
    grpo.add_argument("--output-dir", required=True)
    grpo.add_argument("--tasks-name", default="rl")
    grpo.add_argument("--steps", type=int)
    grpo.add_argument("--group-size", type=int)
    grpo.add_argument("--tasks-per-step", type=int)
    grpo.add_argument("--learning-rate", type=float)
    grpo.add_argument("--max-response-tokens", type=int)
    grpo.add_argument("--max-context-tokens", type=int)
    grpo.add_argument("--kl-coef", type=float)
    grpo.add_argument("--temperature", type=float)
    grpo.add_argument("--seed", type=int)
    grpo.add_argument("--concurrency", type=int)
    grpo.add_argument("--init-adapter", default="")
    grpo.add_argument("--distill-mode", default="forward_kl", choices=["forward_kl", "reverse_kl", "jsd"])
    grpo.add_argument("--distill-beta", type=float, default=0.0)
    grpo.add_argument("--distill-action-tokens-only", action="store_true")
    grpo.add_argument("--teacher-path", default="")
    grpo.add_argument("--self-teacher", action="store_true")
    _add_model_flags(grpo)
    grpo.set_defaults(func=cmd_grpo)

    evaluate = sub.add_parser("eval", help="roll a policy out and score it")
    evaluate.add_argument("--tasks", default="official_test")
    evaluate.add_argument("--out", required=True)
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    evaluate.add_argument("--api-key", default="")
    evaluate.add_argument("--label", default="")
    evaluate.add_argument("--limit", type=int, default=0)
    evaluate.add_argument("--samples", type=int, default=1)
    evaluate.add_argument("--concurrency", type=int, default=4)
    evaluate.add_argument("--temperature", type=float, default=0.0)
    evaluate.add_argument("--max-tokens", type=int, default=1024)
    evaluate.add_argument("--max-turns", type=int, default=30)
    evaluate.add_argument("--keep-tool-results", type=int, default=3)
    evaluate.add_argument("--persona", action="store_true", help="show the shopper's persona on reset")
    evaluate.set_defaults(func=cmd_eval)
    return parser


def _add_model_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn", default="sdpa")
    parser.add_argument("--lora-rank", type=int, default=0)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
