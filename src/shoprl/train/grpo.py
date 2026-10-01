"""Group-relative policy optimisation on verifiable shopping rewards.

The reward is the environment's own score for a completed episode, so no reward
model is trained and there is nothing to drift. Two design points do the real
work here:

* A rollout is worth ``group_size`` candidates on the *same* task, and the
  advantage is the candidate's reward centred inside its own group. Tasks every
  candidate solves, or none solves, carry no gradient, which is exactly right
  and is reported as ``zero_variance_groups``.
* Training samples are turn-level, not trajectory-level: each decision is scored
  against the context the model actually saw, and the group advantage is shared
  across the turns of a candidate. A shopping episode has 20+ turns on one
  sparse reward, so per-turn credit assignment without this split would mean a
  single gradient signal spread over a 16k-token sequence.

Loss form is GRPO with a clipped ratio plus optional KL to the reference and an
optional distillation term (see ``shoprl.train.distill``).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shoprl.data.sft import build_turn_examples
from shoprl.env.catalog import load_catalog
from shoprl.env.local import EnvPool
from shoprl.env.tasks import load_task_pool
from shoprl.env.tools import ACT_SCHEMA, ACT_DESCRIPTION, RESET_DESCRIPTION
from shoprl.harness.context import ContextPolicy
from shoprl.harness.rollout import EpisodeRunner, PROMPT
from shoprl.train.common import (
    ModelConfig,
    load_model,
    load_tokenizer,
    set_seed,
    write_run_manifest,
)
from shoprl.train.distill import DistillConfig, TeacherScorer, distillation_loss
from shoprl.train.rollout import HFRolloutEngine

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "shop_reset",
            "description": RESET_DESCRIPTION,
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {"name": "shop_act", "description": ACT_DESCRIPTION, "parameters": ACT_SCHEMA},
    },
]


def group_advantages(rewards: list[float], *, eps: float = 1e-6) -> list[float]:
    """Centre a candidate group's rewards and scale by their spread.

    A group with no spread returns zeros rather than dividing by a tiny epsilon:
    a task every candidate solves, or none solves, carries no information, and
    turning that into a large arbitrary advantage is how a run optimises noise.

    Pure Python on purpose — this is the part of the objective most likely to be
    subtly wrong, so it is testable without a GPU.
    """
    if not rewards:
        return []
    mean = sum(rewards) / len(rewards)
    variance = sum((value - mean) ** 2 for value in rewards) / len(rewards)
    deviation = variance**0.5
    if deviation == 0:
        return [0.0] * len(rewards)
    return [(value - mean) / (deviation + eps) for value in rewards]


@dataclass(slots=True)
class GrpoConfig:
    model: ModelConfig
    output_dir: Path
    tasks: Path | None = None
    tasks_name: str = "rl"
    steps: int = 100
    group_size: int = 4
    tasks_per_step: int = 5
    max_turns: int = 30
    max_tool_calls: int = 60
    max_context_tokens: int = 16384
    max_response_tokens: int = 256
    keep_tool_results: int = 3
    temperature: float = 1.0
    learning_rate: float = 1e-6
    clip_eps: float = 0.2
    clip_eps_high: float = 0.28
    kl_coef: float = 0.0
    entropy_coef: float = 0.0
    grad_accum: int = 4
    max_seq_len: int = 16384
    concurrency: int = 4
    seed: int = 1234
    save_every: int = 25
    curriculum_easy_first: bool = False
    init_adapter: str = ""
    distill: DistillConfig | None = None
    teacher_path: str = ""
    self_teacher: bool = False
    max_turn_samples_per_trajectory: int = 6


@dataclass(slots=True)
class StepStats:
    step: int
    tasks: int = 0
    rollouts: int = 0
    done_rate: float = 0.0
    mean_reward: float = 0.0
    mean_hard: float = 0.0
    zero_variance_groups: int = 0
    turn_samples: int = 0
    loss: float = 0.0
    wall_seconds: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "tasks": self.tasks,
            "rollouts": self.rollouts,
            "done_rate": round(self.done_rate, 4),
            "mean_reward": round(self.mean_reward, 6),
            "mean_hard": round(self.mean_hard, 6),
            "zero_variance_groups": self.zero_variance_groups,
            "turn_samples": self.turn_samples,
            "loss": round(self.loss, 6),
            "wall_seconds": round(self.wall_seconds, 2),
            **self.extra,
        }


class GrpoTrainer:
    """Owns the policy, the rollout engine, the environment pool and the loop."""

    def __init__(
        self,
        config: GrpoConfig,
        *,
        engine: Any | None = None,
        teacher: TeacherScorer | None = None,
        pool: Any | None = None,
    ):
        import torch

        self.config = config
        self.torch = torch
        set_seed(config.seed)
        self.tokenizer = load_tokenizer(config.model)
        self.model = load_model(
            config.model,
            trainable=True,
            adapter=config.init_adapter or None,
        )
        self.engine = engine or HFRolloutEngine(
            self.model, self.tokenizer, max_new_tokens=config.max_response_tokens
        )
        self.teacher = teacher
        self.pool = pool or self._default_pool()
        self.optimizer = torch.optim.AdamW(
            [parameter for parameter in self.model.parameters() if parameter.requires_grad],
            lr=config.learning_rate,
            betas=(0.9, 0.99),
        )
        self.runner = EpisodeRunner(
            self.engine,
            self.pool,
            context_policy=ContextPolicy(
                keep_recent_tool_results=config.keep_tool_results,
                max_prompt_tokens=config.max_context_tokens,
            ),
            max_turns=config.max_turns,
            max_tool_calls=config.max_tool_calls,
            temperature=config.temperature,
            max_tokens=config.max_response_tokens,
            prompt=PROMPT,
        )

    def _default_pool(self) -> EnvPool:
        catalog = load_catalog()
        from shoprl.env.search import Bm25Index

        index = Bm25Index.build(catalog)
        return EnvPool(catalog, index, capacity=max(1, self.config.concurrency))

    # ── rollout ───────────────────────────────────────────────────────

    def _rollout(self, task_ids: list[int]) -> dict[int, list[Any]]:
        """``group_size`` candidates per task, generated without gradient."""
        self.model.eval()
        groups: dict[int, list[Any]] = {}
        for task_id in task_ids:
            candidates = []
            for sample in range(self.config.group_size):
                trajectory = self.runner.run(
                    task_id,
                    session_id=f"s{self.config.seed}-{task_id}-{sample}-{int(time.time() * 1000)}",
                )
                candidates.append(trajectory)
            groups[task_id] = candidates
        self.model.train()
        return groups

    def _turn_samples(self, trajectory: Any) -> list[dict[str, Any]]:
        """Turn-level training samples, keeping only turns with matching tokens."""
        record = trajectory.to_json()
        examples, _rejected = build_turn_examples(
            record, self.tokenizer, tools=TOOLS, max_tokens=self.config.max_seq_len
        )
        samples = []
        for example, target in zip(examples, record["targets"]):
            sampled = target.get("token_ids")
            old_logprobs = target.get("logprobs")
            target_span = example.labels[len(example.input_ids) - len(sampled or []) :] if sampled else []
            if not sampled or len(sampled) != len(target_span) or sampled != target_span:
                continue
            if not old_logprobs or len(old_logprobs) != len(sampled):
                continue
            samples.append(
                {
                    "input_ids": example.input_ids,
                    "loss_mask": [0] * (len(example.input_ids) - len(sampled)) + [1] * len(sampled),
                    "old_logprobs": [0.0] * (len(example.input_ids) - len(sampled)) + list(old_logprobs),
                }
            )
        if self.config.max_turn_samples_per_trajectory:
            samples = samples[-self.config.max_turn_samples_per_trajectory :]
        return samples

    # ── objective ─────────────────────────────────────────────────────

    def _logprobs(self, input_ids: list[int], loss_mask: list[int]):
        """Per-position log-probabilities of the given tokens under the policy."""
        torch = self.torch
        tensor = torch.tensor([input_ids], device=self.model.device)
        logits = self.model(tensor).logits.float()[:, :-1]
        logprobs = torch.log_softmax(logits, dim=-1)
        targets = tensor[:, 1:].unsqueeze(-1)
        gathered = logprobs.gather(-1, targets).squeeze(-1)[0]
        mask = torch.tensor([loss_mask], device=self.model.device, dtype=torch.bool)[:, 1:]
        return gathered, mask

    def _policy_loss(self, sample: dict[str, Any], advantage: float):
        torch = self.torch
        current, mask = self._logprobs(sample["input_ids"], sample["loss_mask"])
        old = torch.tensor(
            [sample["old_logprobs"]], device=self.model.device, dtype=current.dtype
        )[:, 1:]
        advantage_tensor = torch.full_like(current, advantage)

        ratio = (current - old).exp()
        unclipped = ratio * advantage_tensor
        clipped = ratio.clamp(1 - self.config.clip_eps, 1 + self.config.clip_eps_high) * advantage_tensor
        per_token = -torch.min(unclipped, clipped)

        if self.config.kl_coef > 0:
            with self._reference_weights():
                reference, _ = self._logprobs(sample["input_ids"], sample["loss_mask"])
            per_token = per_token + self.config.kl_coef * (current - reference)
        if self.config.entropy_coef > 0:
            per_token = per_token - self.config.entropy_coef * current

        total = (per_token * mask).sum() / mask.sum().clamp_min(1)
        if self.teacher is not None and self.config.distill is not None:
            teacher_logprobs = self.teacher.score(sample["input_ids"])
            teacher_tensor = torch.tensor(
                [teacher_logprobs], device=self.model.device, dtype=current.dtype
            )[:, 1:]
            total = total + distillation_loss(
                current, teacher_tensor, mask, self.config.distill
            )
        return total

    def _reference_weights(self):
        """Reference policy for the KL term: the base weights under a LoRA adapter."""
        if self.config.model.lora_rank > 0 and hasattr(self.model, "disable_adapter"):
            return self.model.disable_adapter()
        import contextlib

        return contextlib.nullcontext()

    # ── loop ──────────────────────────────────────────────────────────

    def train(self) -> list[dict[str, Any]]:
        tasks = self._load_tasks()
        history: list[StepStats] = []
        for step in range(self.config.steps):
            started = time.perf_counter()
            batch = self._next_tasks(tasks, step)
            groups = self._rollout(batch)
            stats, gradients = self._advantages(groups)
            stats.step = step
            loss = self._update(gradients)
            stats.loss = loss
            stats.wall_seconds = time.perf_counter() - started
            history.append(stats)
            self._log(stats)
            if self.config.save_every and (step + 1) % self.config.save_every == 0:
                self.save(step)
        self.save(self.config.steps)
        return [item.to_json() for item in history]

    def _load_tasks(self) -> list[Any]:
        tasks = load_task_pool(self.config.tasks_name, self.config.tasks)
        if self.config.curriculum_easy_first:
            tasks = sorted(tasks, key=lambda task: (task.difficulty, task.task_id))
        return tasks

    def _next_tasks(self, tasks: list[Any], step: int) -> list[int]:
        import random

        rng = random.Random(self.config.seed + step)
        count = min(self.config.tasks_per_step, len(tasks))
        return [task.task_id for task in rng.sample(tasks, count)]

    def _advantages(self, groups: dict[int, list[Any]]) -> tuple[StepStats, list[tuple[dict[str, Any], float]]]:
        torch = self.torch
        stats = StepStats(step=0)
        gradients: list[tuple[dict[str, Any], float]] = []
        rewards: list[float] = []

        for candidates in groups.values():
            stats.tasks += 1
            stats.rollouts += len(candidates)
            group_rewards = [float(candidate.reward) for candidate in candidates]
            rewards.extend(group_rewards)
            advantages = group_advantages(group_rewards)
            if not any(advantages):
                stats.zero_variance_groups += 1
                continue
            for candidate, advantage in zip(candidates, advantages):
                for sample in self._turn_samples(candidate):
                    gradients.append((sample, advantage))
                    stats.turn_samples += 1

        stats.mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
        stats.mean_hard = (
            sum(float(c.reward_detail.get("r_hard", 0.0)) for group in groups.values() for c in group)
            / max(1, sum(len(group) for group in groups.values()))
        )
        stats.done_rate = (
            sum(1 for group in groups.values() for c in group if c.termination == "done")
            / max(1, sum(len(group) for group in groups.values()))
        )
        return stats, gradients

    def _update(self, gradients: list[tuple[dict[str, Any], float]]) -> float:
        """Accumulate the turn-level losses and take one optimiser step."""
        if not gradients:
            return 0.0
        self.optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for index, (sample, advantage) in enumerate(gradients):
            loss = self._policy_loss(sample, advantage) / len(gradients)
            loss.backward()
            total += float(loss.detach())
        self.torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], 1.0
        )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return total

    def save(self, step: int) -> Path:
        path = Path(self.config.output_dir) / f"step-{step}"
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path))
        self.tokenizer.save_pretrained(str(path))
        write_run_manifest(
            path,
            {
                "step": step,
                "base_model": self.config.model.name_or_path,
                "lora_rank": self.config.model.lora_rank,
                "group_size": self.config.group_size,
                "tasks_per_step": self.config.tasks_per_step,
                "learning_rate": self.config.learning_rate,
                "seed": self.config.seed,
                "distill": self.config.distill.__dict__ if self.config.distill else None,
            },
        )
        return path

    def _log(self, stats: StepStats) -> None:
        line = json.dumps(stats.to_json(), ensure_ascii=False)
        print(line, flush=True)
        path = Path(self.config.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        with (path / "train_log.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
