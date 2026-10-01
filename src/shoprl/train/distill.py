"""On-policy distillation terms that ride along with the RL objective.

The student generates the rollout, so the teacher is always scored on tokens the
student actually produced. Two knobs matter and they are separate:

* ``mode`` picks the divergence. Forward KL is mass-covering and is the safer
  default when the teacher is much stronger; reverse KL is mode-seeking and
  sharpens an already competent student.
* ``weights`` decides which tokens count. Distilling the whole turn also teaches
  the model to imitate the teacher's prose; restricting the term to the action
  tokens keeps the gradient on the decision that earned the reward.

Both a teacher-only term (OPD) and a self-teacher with privileged context (OPSD)
use this module; only the scorer differs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

Mode = str


@dataclass(slots=True)
class DistillConfig:
    mode: Mode = "forward_kl"
    beta: float = 0.1
    temperature: float = 1.0
    action_tokens_only: bool = False


class TeacherScorer(Protocol):
    """Scores a fixed token sequence; must not resample."""

    def score(self, input_ids: Sequence[int], *, context: list[dict[str, Any]] | None = None) -> list[float]:
        """Per-position log-probabilities of ``input_ids`` under the teacher."""
        ...


def forward_kl(student: Any, teacher: Any) -> Any:
    """``KL(teacher || student)`` per position."""
    return teacher.exp() * (teacher - student)


def reverse_kl(student: Any, teacher: Any) -> Any:
    """``KL(student || teacher)`` per position."""
    return student.exp() * (student - teacher)


def jsd(student: Any, teacher: Any, beta: float = 0.5) -> Any:
    """Jensen-Shannon divergence, a bounded compromise between the two KLs."""

    mix = (student.exp() * (1 - beta) + teacher.exp() * beta).clamp_min(1e-12).log()
    return (1 - beta) * forward_kl(student, mix) + beta * forward_kl(teacher, mix)


LOSSES = {"forward_kl": forward_kl, "reverse_kl": reverse_kl, "jsd": jsd}


def distillation_loss(
    student_logprobs: Any,
    teacher_logprobs: Any,
    loss_mask: Any,
    config: DistillConfig,
) -> Any:
    """Masked distillation term; ``loss_mask`` marks the positions that count."""

    per_position = LOSSES[config.mode](student_logprobs, teacher_logprobs)
    weights = loss_mask.to(per_position.dtype)
    total = (per_position * weights).sum()
    return total / weights.sum().clamp_min(1.0) * config.beta


class LocalTeacherScorer:
    """Teacher log-probabilities from a local model, optionally with extra context.

    Passing ``context`` implements self-distillation: the same weights see a
    privileged prompt (for example the task's goal options) that the student did
    not have, so the teacher signal carries information rather than just echoing
    the student.
    """

    def __init__(self, model: Any, tokenizer: Any, *, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def score(self, input_ids: Sequence[int], *, context: list[dict[str, Any]] | None = None) -> list[float]:
        import torch

        tokens = list(input_ids)
        prefix: list[int] = []
        if context:
            prefix = list(
                self.tokenizer.apply_chat_template(
                    context, tokenize=True, add_generation_prompt=True
                )
            )
        tensor = torch.tensor([prefix + tokens], device=self.device)
        with torch.inference_mode():
            logits = self.model(tensor).logits.float()
        logprobs = torch.log_softmax(logits[:, :-1], dim=-1)
        targets = tensor[:, 1:].unsqueeze(-1)
        gathered = logprobs.gather(-1, targets).squeeze(-1)[0]
        return [float(value) for value in gathered[len(prefix) :]]

