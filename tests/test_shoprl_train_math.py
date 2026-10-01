"""The RL objective's arithmetic.

Everything that can be checked without a GPU is checked here; the tensor-level
pieces are covered when torch is importable, which is the case on the training
host and not on the machine this repository was written on.
"""

from __future__ import annotations

import pytest

from shoprl.train.grpo import group_advantages


def test_advantages_are_centred_within_the_group():
    advantages = group_advantages([1.0, 0.5, 0.5, 0.0])

    assert sum(advantages) == pytest.approx(0.0, abs=1e-9)
    assert advantages[0] > 0 > advantages[-1]
    assert advantages[1] == advantages[2]


def test_a_group_with_no_spread_carries_no_gradient():
    """The whole point of group-relative advantages: a saturated task is skipped."""
    assert group_advantages([1.0, 1.0, 1.0]) == [0.0, 0.0, 0.0]
    assert group_advantages([0.0, 0.0]) == [0.0, 0.0]
    assert group_advantages([]) == []


def test_advantages_are_scale_free():
    """Only the ordering within a group matters, so the pool's scale does not."""
    tight = group_advantages([0.02, 0.01, 0.0])
    wide = group_advantages([1.0, 0.5, 0.0])

    # Approximate, not exact: the epsilon guard in the denominator is a fixed
    # constant, so it weighs slightly more on a tightly clustered group.
    assert tight == pytest.approx(wide, rel=1e-3)


def test_the_worst_candidate_gets_a_negative_advantage():
    advantages = group_advantages([0.0, 0.3, 0.6])
    assert advantages[0] < 0 < advantages[-1]


def test_distillation_losses_are_finite_and_minimised_at_agreement():
    torch = pytest.importorskip("torch")
    from shoprl.train.distill import DistillConfig, distillation_loss

    student = torch.log_softmax(torch.randn(1, 8), dim=-1)
    teacher = torch.log_softmax(torch.randn(1, 8), dim=-1)
    mask = torch.ones(1, 8)
    config = DistillConfig(beta=0.5)

    apart = distillation_loss(student, teacher, mask, config)
    together = distillation_loss(teacher, teacher, mask, config)

    assert torch.isfinite(apart) and apart >= 0
    assert together == pytest.approx(0.0, abs=1e-6)
    assert apart > together


def test_distillation_respects_the_loss_mask():
    torch = pytest.importorskip("torch")
    from shoprl.train.distill import DistillConfig, distillation_loss

    student = torch.log_softmax(torch.randn(1, 4), dim=-1)
    teacher = torch.log_softmax(torch.randn(1, 4), dim=-1)
    config = DistillConfig(beta=1.0)

    masked_out = distillation_loss(student, teacher, torch.zeros(1, 4), config)

    assert torch.isfinite(masked_out)
    assert masked_out == pytest.approx(0.0, abs=1e-6)
