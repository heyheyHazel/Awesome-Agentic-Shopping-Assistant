"""Percentile-based RFM scoring and customer segmentation."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable
from shopping_assistant.catalog import USERS, DemoUser, get_user
from shopping_assistant.domain.models import RFM, UserProfile

SEGMENTS = ["Champions", "Loyal", "Potential", "At Risk", "New"]

# Composite score weights: recency 30%, frequency 30%, monetary 40%.
RECENCY_WEIGHT, FREQUENCY_WEIGHT, MONETARY_WEIGHT = 0.3, 0.3, 0.4

# Segment cut-offs, expressed as percentile standing in the shopper base.
NEW_CUTOFF = 0.30       # bottom 30% by order count — still ramping up
DORMANT_CUTOFF = 0.33   # bottom third by recency — has not bought in a while
HIGH_CUTOFF = 0.66
MID_CUTOFF = 0.50


class RFMScale:
    """Percentile scale fitted on a shopper population.

    RFM standings are relative by definition — 20 orders makes a heavy buyer in a
    base whose median is 4 and a light one where the median is 40 — so scores are
    percentile ranks against the loaded shopper base rather than fixed divisors.
    That keeps every segment populated whatever the catalog or shopper base is,
    which fixed thresholds cannot do once the data stops looking like the four
    hand-written demo shoppers they were originally tuned on.
    """

    def __init__(self, users: Iterable[DemoUser]):
        self._recency = sorted(user.recency_days for user in users)
        self._frequency = sorted(user.orders for user in users)
        self._monetary = sorted(user.lifetime_value for user in users)

    @staticmethod
    def _percentile(sorted_values: list, value: float) -> float:
        """Share of the population at or below `value`, in [0, 1]."""
        if not sorted_values:
            return 0.0
        return bisect_right(sorted_values, value) / len(sorted_values)

    def scores(self, user: DemoUser) -> tuple[float, float, float]:
        """Return (recency, frequency, monetary) percentile scores."""
        frequency = self._percentile(self._frequency, user.orders)
        monetary = self._percentile(self._monetary, user.lifetime_value)
        # Fewer days since the last purchase is better, so recency is inverted.
        recency = 1.0 - self._percentile(self._recency, user.recency_days)
        return recency, frequency, monetary


SCALE = RFMScale(USERS.values())


def compute_rfm(user: DemoUser, scale: RFMScale = SCALE) -> RFM:
    """Score one shopper against their customer base."""
    recency, frequency, monetary = scale.scores(user)
    return RFM(
        recency_days=user.recency_days,
        orders=user.orders,
        lifetime_value=user.lifetime_value,
        recency=round(recency, 3),
        frequency=round(frequency, 3),
        monetary=round(monetary, 3),
        overall=round(
            RECENCY_WEIGHT * recency + FREQUENCY_WEIGHT * frequency + MONETARY_WEIGHT * monetary,
            3,
        ),
    )


def classify(user: DemoUser, scale: RFMScale = SCALE) -> str:
    """Assign one of the five segments from the shopper's percentile standing.

    Order matters: the positive signals are tested before the dormancy rule, so a
    heavy buyer who has gone quiet is still recognised as Loyal rather than losing
    their segment to "At Risk".
    """
    scores = compute_rfm(user, scale)
    if scores.frequency <= NEW_CUTOFF:
        return "New"
    if (
        scores.recency >= MID_CUTOFF
        and scores.frequency >= HIGH_CUTOFF
        and scores.monetary >= HIGH_CUTOFF
    ):
        return "Champions"
    if scores.frequency >= HIGH_CUTOFF and scores.monetary >= MID_CUTOFF:
        return "Loyal"
    if scores.recency <= DORMANT_CUTOFF:
        return "At Risk"
    return "Potential"


def build_profile(user_id: str) -> UserProfile | None:
    """Load a shopper and attach their RFM scores and segment."""
    user = get_user(user_id)
    if user is None:
        return None
    return UserProfile(
        user_id=user.user_id,
        name=user.name,
        email=user.email,
        tier=user.tier,
        segment=classify(user),
        rfm=compute_rfm(user),
        preferred_categories=user.preferred_categories,
        price_range=user.price_range,
    )
