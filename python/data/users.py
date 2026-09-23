"""Built-in demo shoppers with pre-built behaviour histories (used for RFM profiling).

Money is in CNY. This file is the fallback used when `data/generated/` is absent
(see data/store.py); the four shoppers cover four of the five RFM segments.
"""

from dataclasses import dataclass, field


@dataclass
class DemoUser:
    user_id: str
    name: str
    email: str
    tier: str  # VIP | Standard
    recency_days: int  # days since last purchase
    orders: int  # purchases in the last 180 days
    lifetime_value: float  # total spend in CNY
    preferred_categories: list[str] = field(default_factory=list)
    price_range: tuple[float, float] = (0.0, 5000.0)


MOCK_USERS: dict[str, DemoUser] = {
    "U001": DemoUser(
        user_id="U001", name="Emily Johnson", email="emily@example.com", tier="VIP",
        recency_days=5, orders=12, lifetime_value=8800.00,
        preferred_categories=["跑鞋", "美妆", "护肤"],
        price_range=(400.0, 2000.0),
    ),
    "U002": DemoUser(
        user_id="U002", name="Michael Chen", email="michael@example.com", tier="Standard",
        recency_days=2, orders=1, lifetime_value=629.93,
        preferred_categories=["休闲鞋", "运动"],
        price_range=(200.0, 900.0),
    ),
    "U003": DemoUser(
        user_id="U003", name="Sarah Kim", email="sarah@example.com", tier="Standard",
        recency_days=21, orders=8, lifetime_value=4483.50,
        preferred_categories=["美妆", "服饰", "家居"],
        price_range=(150.0, 1400.0),
    ),
    "U004": DemoUser(
        user_id="U004", name="David Park", email="david@example.com", tier="Standard",
        recency_days=75, orders=3, lifetime_value=1472.80,
        preferred_categories=["音频", "配饰"],
        price_range=(300.0, 1800.0),
    ),
}
