"""A/B testing engine: consistent hashing assignment plus Thompson Sampling."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from models.schemas import ExperimentInfo, VariantStats


@dataclass
class Variant:
    name: str
    label: str
    successes: int = 1
    failures: int = 1

    @property
    def trials(self) -> int:
        return self.successes + self.failures

    @property
    def conversion_rate(self) -> float:
        return self.successes / self.trials


@dataclass
class Experiment:
    experiment_id: str
    name: str
    variants: list[Variant] = field(default_factory=list)

    @property
    def winner(self) -> Variant:
        return max(self.variants, key=lambda v: v.conversion_rate)


class ABTestEngine:
    """Seeded with demo trial data; assignments are stable per user."""

    def __init__(self):
        self.experiment = Experiment(
            experiment_id="layout_variant",
            name="Layout Variant",
            variants=[
                Variant(name="A", label="Current", successes=313, failures=186),
                Variant(name="B", label="New", successes=370, failures=128),
            ],
        )

    def assign(self, user_id: str) -> str:
        """Deterministically assign a user to a variant by consistent hashing."""
        digest = hashlib.md5(f"{user_id}:{self.experiment.experiment_id}".encode()).hexdigest()
        bucket = int(digest[:8], 16)
        return self.experiment.variants[bucket % len(self.experiment.variants)].name

    def sample(self) -> str:
        """Thompson Sampling: draw from the Beta posteriors and pick the best arm."""
        draws = [(np.random.beta(v.successes, v.failures), v.name) for v in self.experiment.variants]
        return max(draws)[1]

    def record_outcome(self, variant_name: str, success: bool) -> None:
        """Update the Beta posterior of one variant with an observed outcome."""
        for variant in self.experiment.variants:
            if variant.name == variant_name:
                if success:
                    variant.successes += 1
                else:
                    variant.failures += 1
                return

    def info(self, user_id: str | None = None) -> ExperimentInfo:
        """Current experiment state, including the variant of a given user."""
        return ExperimentInfo(
            experiment_id=self.experiment.experiment_id,
            name=self.experiment.name,
            variant=self.assign(user_id) if user_id else None,
            variants=[
                VariantStats(
                    name=v.name,
                    label=v.label,
                    trials=v.trials,
                    conversion_rate=round(v.conversion_rate * 100, 1),
                )
                for v in self.experiment.variants
            ],
            winner=self.experiment.winner.name,
        )
