"""Demo shoppers with pre-built behavior histories (used for RFM profiling)."""

from dataclasses import dataclass, field


@dataclass
class DemoUser:
    user_id: str
    name: str
    email: str
    tier: str  # VIP | Standard
    recency_days: int  # days since last purchase
    orders: int  # purchases in the last 180 days
    lifetime_value: float  # total spend in USD
    preferred_categories: list[str] = field(default_factory=list)
    price_range: tuple[float, float] = (0.0, 500.0)


USERS: dict[str, DemoUser] = {
    "U001": DemoUser(
        user_id="U001", name="Emily Johnson", email="emily@example.com", tier="VIP",
        recency_days=5, orders=12, lifetime_value=1256.00,
        preferred_categories=["Running Shoes", "Beauty", "Skincare"],
        price_range=(50.0, 250.0),
    ),
    "U002": DemoUser(
        user_id="U002", name="Michael Chen", email="michael@example.com", tier="Standard",
        recency_days=2, orders=1, lifetime_value=89.99,
        preferred_categories=["Sneakers", "Sports"],
        price_range=(30.0, 120.0),
    ),
    "U003": DemoUser(
        user_id="U003", name="Sarah Kim", email="sarah@example.com", tier="Standard",
        recency_days=21, orders=8, lifetime_value=640.50,
        preferred_categories=["Beauty", "Fashion", "Home"],
        price_range=(20.0, 200.0),
    ),
    "U004": DemoUser(
        user_id="U004", name="David Park", email="david@example.com", tier="Standard",
        recency_days=75, orders=3, lifetime_value=210.40,
        preferred_categories=["Audio", "Accessories"],
        price_range=(40.0, 260.0),
    ),
}

DEFAULT_USER_ID = "U001"


def get_user(user_id: str) -> DemoUser:
    return USERS.get(user_id, USERS[DEFAULT_USER_ID])
