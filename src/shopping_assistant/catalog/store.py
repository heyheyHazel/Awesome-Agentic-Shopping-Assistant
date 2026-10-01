"""Catalog and shopper data source.

Real data comes from `scripts/fetch_data.py`, which converts the ShopSimulator
dataset into `data/generated/*.json`. Those files are NOT committed (the upstream
dataset ships without a license), so a fresh clone falls back to the small
built-in demo catalog and the app still runs end to end.

`ECOM_DATA_SOURCE` selects the behaviour:

    auto (default)  use generated data when present, otherwise the mock catalog
    real            require generated data, fail fast when it is missing
    mock            always use the mock catalog (tests rely on stable product IDs)
"""

from __future__ import annotations

import json

import structlog

from shopping_assistant.catalog.demo_products import MOCK_PRODUCTS
from shopping_assistant.catalog.demo_users import MOCK_USERS, DemoUser
from shopping_assistant.domain.models import Product
from shopping_assistant.settings import get_settings

logger = structlog.get_logger()

GENERATED_DIR = get_settings().data_dir / "generated"
CATALOG_FILE = GENERATED_DIR / "catalog.json"
SHOPPERS_FILE = GENERATED_DIR / "shoppers.json"

MOCK_DEFAULT_USER_ID = "U001"


def _to_product(record: dict) -> Product:
    return Product(
        product_id=str(record["product_id"]),
        name=str(record["name"]),
        category=str(record.get("category", "")),
        price=float(record["price"]),
        brand=str(record.get("brand", "")),
        stock=int(record.get("stock", 0)),
        rating=float(record.get("rating", 0.0)),
        rating_count=int(record.get("rating_count", 0)),
        tags=[str(tag) for tag in record.get("tags") or []],
    )


def _to_shopper(record: dict) -> DemoUser:
    band = list(record.get("price_range") or [])[:2] or [0.0, 0.0]
    low, high = float(band[0]), float(band[1])
    return DemoUser(
        user_id=str(record["user_id"]),
        name=str(record.get("name") or record["user_id"]),
        email=str(record.get("email", "")),
        tier=str(record.get("tier", "Standard")),
        recency_days=int(record.get("recency_days", 0)),
        orders=int(record.get("orders", 0)),
        lifetime_value=float(record.get("lifetime_value", 0.0)),
        preferred_categories=[str(c) for c in record.get("preferred_categories") or []],
        price_range=(low, max(high, low)),
    )


def _read_generated() -> tuple[list[Product], dict[str, DemoUser]] | None:
    """Load the converted dataset, or None when it has not been fetched yet."""
    if not (CATALOG_FILE.exists() and SHOPPERS_FILE.exists()):
        return None

    catalog = json.loads(CATALOG_FILE.read_text(encoding="utf-8"))
    shoppers = json.loads(SHOPPERS_FILE.read_text(encoding="utf-8"))

    products = [_to_product(record) for record in catalog]
    users = {record["user_id"]: _to_shopper(record) for record in shoppers}
    return (products, users) if products and users else None


def _resolve() -> tuple[list[Product], dict[str, DemoUser], str]:
    source = get_settings().data_source
    if source in ("auto", "real"):
        try:
            loaded = _read_generated()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning("data.generated_unreadable", error=str(exc))
            loaded = None

        if loaded:
            products, users = loaded
            logger.info("data.source", source="shopsimulator", products=len(products), shoppers=len(users))
            return products, users, "shopsimulator"

        if source == "real":
            raise RuntimeError(
                "ECOM_DATA_SOURCE=real but no usable data in "
                f"{GENERATED_DIR}. Run: python scripts/fetch_data.py"
            )

    logger.info("data.source", source="mock", products=len(MOCK_PRODUCTS), shoppers=len(MOCK_USERS))
    return list(MOCK_PRODUCTS), dict(MOCK_USERS), "mock"


PRODUCTS, USERS, SOURCE = _resolve()

# Mock IDs are U001.. ; ShopSimulator personas are U08…, so prefer the mock
# default when it exists and otherwise take the first real shopper.
DEFAULT_USER_ID = MOCK_DEFAULT_USER_ID if MOCK_DEFAULT_USER_ID in USERS else min(USERS)


def get_user(user_id: str) -> DemoUser:
    """Look up a shopper, falling back to the default demo user."""
    return USERS.get(user_id, USERS[DEFAULT_USER_ID])
