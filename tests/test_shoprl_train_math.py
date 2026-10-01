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


# ── the GRPO objective ────────────────────────────────────────────────
# These run wherever torch is importable, which includes the CPU CI job and any
# training host. They are the only check on the objective itself: no reward, no
# environment and no GPU are involved, just the arithmetic a run would optimise.


def test_token_logprobs_match_a_manual_log_softmax():
    torch = pytest.importorskip("torch")
    from shoprl.train.grpo import gather_token_logprobs

    torch.manual_seed(0)
    logits = torch.randn(1, 5, 7)
    ids = torch.tensor([[1, 2, 3, 4, 5]])

    got = gather_token_logprobs(logits, ids)[0]
    expected = torch.log_softmax(logits[:, :-1], dim=-1)[0, range(4), ids[0, 1:]]

    assert got.shape == (4,)
    assert torch.allclose(got, expected)


def test_an_unchanged_policy_scores_the_plain_advantage():
    torch = pytest.importorskip("torch")
    from shoprl.train.grpo import clipped_policy_loss

    same = torch.tensor([[-1.0, -2.0, -0.5]])
    loss = clipped_policy_loss(same, same, 0.5, torch.ones(1, 3))

    # ratio == 1, so the surrogate is exactly -advantage.
    assert float(loss) == pytest.approx(-0.5)


def test_clipping_stops_a_large_update_from_paying_off():
    torch = pytest.importorskip("torch")
    from shoprl.train.grpo import clipped_policy_loss

    old = torch.zeros(1, 1)
    mask = torch.ones(1, 1)

    far_above = clipped_policy_loss(
        torch.full((1, 1), 5.0), old, 1.0, mask, clip_low=0.2, clip_high=0.28
    )
    far_below = clipped_policy_loss(torch.full((1, 1), -5.0), old, -1.0, mask)

    # Upward: min(ratio*A, clipped*A) caps the reward at the high clip.
    assert float(far_above) == pytest.approx(-1.28)
    # Downward: the clipped branch is the smaller one, and it floors the loss at
    # the low clip rather than letting a ruined ratio escape the penalty.
    assert float(far_below) == pytest.approx(0.8)


def test_masked_tokens_carry_no_gradient_signal():
    torch = pytest.importorskip("torch")
    from shoprl.train.grpo import clipped_policy_loss

    current = torch.tensor([[-1.0, 4.0]])
    old = torch.zeros(1, 2)

    only_first = clipped_policy_loss(current, old, 1.0, torch.tensor([[True, False]]))
    also_second = clipped_policy_loss(current, old, 1.0, torch.tensor([[True, True]]))
    shifted = clipped_policy_loss(current, torch.tensor([[0.0, -9.0]]), 1.0, torch.tensor([[True, False]]))

    assert float(only_first) != float(also_second)
    # Changing a masked-out token's old logprob changes nothing.
    assert float(shifted) == pytest.approx(float(only_first))


def test_an_all_masked_batch_is_zero_not_undefined():
    torch = pytest.importorskip("torch")
    from shoprl.train.grpo import clipped_policy_loss, k3_kl

    zeros = torch.zeros(1, 3)
    mask = torch.zeros(1, 3, dtype=torch.bool)

    assert float(clipped_policy_loss(zeros, zeros, 1.0, mask)) == pytest.approx(0.0)
    assert float(k3_kl(zeros, zeros + 1.0, mask)) == pytest.approx(0.0)


def test_kl_is_zero_at_agreement_and_matches_the_closed_form_at_a_known_offset():
    torch = pytest.importorskip("torch")
    from shoprl.train.grpo import k3_kl

    current = torch.zeros(1, 2)
    mask = torch.ones(1, 2)

    assert float(k3_kl(current, current.clone(), mask)) == pytest.approx(0.0, abs=1e-7)
    delta = 0.3
    expected = torch.exp(torch.tensor(delta)) - delta - 1
    assert float(k3_kl(current, current + delta, mask)) == pytest.approx(float(expected))


def test_the_trainer_objective_runs_through_a_real_forward_path():
    """White-box on purpose: this is the composition that has never executed."""
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from shoprl.train.grpo import GrpoTrainer, gather_token_logprobs

    class StubModel:
        device = "cpu"

        def __init__(self, logits):
            self._logits = logits

        def __call__(self, input_ids):
            return SimpleNamespace(logits=self._logits)

    input_ids = torch.tensor([[1, 2, 3, 4]])
    logits = torch.randn(1, 4, 9)

    trainer = GrpoTrainer.__new__(GrpoTrainer)
    trainer.torch = torch
    trainer.model = StubModel(logits)
    trainer.teacher = None
    trainer.config = SimpleNamespace(
        clip_eps=0.2, clip_eps_high=0.28, kl_coef=0.0, entropy_coef=0.0,
        model=SimpleNamespace(lora_rank=0), distill=None,
    )
    # The recorded logprobs are the policy's own, so the ratio is exactly 1.
    sample = {
        "input_ids": input_ids[0].tolist(),
        "loss_mask": [0, 1, 1, 1],
        "old_logprobs": [0.0] + gather_token_logprobs(logits, input_ids)[0].tolist(),
    }

    loss = trainer._policy_loss(sample, advantage=0.75)

    assert float(loss) == pytest.approx(-0.75)
