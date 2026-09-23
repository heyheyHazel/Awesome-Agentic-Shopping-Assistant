"""A/B engine: stable assignment, conversion stats, and Thompson Sampling."""

from services.ab_test import ABTestEngine


def test_assignment_is_stable_per_user():
    engine = ABTestEngine()
    assert engine.assign("U001") == engine.assign("U001")
    assert engine.assign("U001") in {"A", "B"}


def test_info_reports_seeded_conversion_rates():
    info = ABTestEngine().info("U001")
    rates = {v.name: v.conversion_rate for v in info.variants}
    assert rates["A"] == 62.7
    assert rates["B"] == 74.3
    assert info.winner == "B"
    assert info.variant in {"A", "B"}


def test_thompson_sampling_prefers_the_winning_arm():
    engine = ABTestEngine()
    wins = sum(1 for _ in range(200) if engine.sample() == "B")
    assert wins > 120


def test_record_outcome_updates_posterior():
    engine = ABTestEngine()
    before = engine.info().variants[1].trials
    engine.record_outcome("B", success=True)
    assert engine.info().variants[1].trials == before + 1
