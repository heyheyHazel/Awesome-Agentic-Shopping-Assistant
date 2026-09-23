"""User profile agent: computes RFM scores and a customer segment from behavior data."""

from __future__ import annotations

from typing import Any

from config import get_settings
from data.users import DemoUser, get_user
from models.schemas import RFM, UserProfile, UserProfileResult

from .base_agent import BaseAgent

SEGMENTS = ["Champions", "Loyal", "Potential", "At Risk", "New"]


def compute_rfm(user: DemoUser) -> RFM:
    """Normalise raw behavior into 0-1 scores: 30% recency, 30% frequency, 40% monetary."""
    recency = max(0.0, 1.0 - user.recency_days / 90)
    frequency = min(1.0, user.orders / 12)
    monetary = min(1.0, user.lifetime_value / 1500)
    return RFM(
        recency_days=user.recency_days,
        orders=user.orders,
        lifetime_value=user.lifetime_value,
        recency=round(recency, 3),
        frequency=round(frequency, 3),
        monetary=round(monetary, 3),
        overall=round(0.3 * recency + 0.3 * frequency + 0.4 * monetary, 3),
    )


def classify(user: DemoUser) -> str:
    """Rule-based RFM segment assignment."""
    if user.orders <= 2:
        return "New"
    if user.recency_days > 60:
        return "At Risk"
    if user.orders >= 10 and user.lifetime_value >= 1000 and user.recency_days <= 14:
        return "Champions"
    if user.orders >= 5 and user.recency_days <= 45:
        return "Loyal"
    return "Potential"


class UserProfileAgent(BaseAgent):
    """Builds the shopper profile consumed by re-ranking and copy generation."""

    def __init__(self):
        super().__init__(name="profile", timeout=get_settings().agent_timeout_default)

    async def _execute(self, user_id: str, **_: Any) -> UserProfileResult:
        """Load the user and attach RFM scores and segment."""
        user = get_user(user_id)
        profile = UserProfile(
            user_id=user.user_id,
            name=user.name,
            email=user.email,
            tier=user.tier,
            segment=classify(user),
            rfm=compute_rfm(user),
            preferred_categories=user.preferred_categories,
            price_range=user.price_range,
        )
        return UserProfileResult(profile=profile)
