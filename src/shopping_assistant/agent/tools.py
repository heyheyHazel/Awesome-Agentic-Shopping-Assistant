"""Tools the shopping agent can call.

Every tool is deterministic: the model decides *what* to look up and *which*
products to present, but retrieval, the stock check and constraint enforcement
are code.

Payloads the UI renders (product cards, profile, inventory) are emitted from the
tool's own result rather than from the model's wording, so a stated budget or a
sold-out item cannot drift with the prose.

The functions are plain and stateless over the loaded catalog, which keeps them
directly testable, and they are declared to the harness as JSON schemas produced
by :func:`build_tool_registry`.
"""

from __future__ import annotations

from typing import Any

from shoprl.harness.tools import ToolRegistry

from shopping_assistant.catalog import PRODUCTS
from shopping_assistant.domain.currency import currency_symbol
from shopping_assistant.domain.inventory import check_stock
from shopping_assistant.domain.models import Product, SearchParams
from shopping_assistant.domain.rfm import build_profile
from shopping_assistant.retrieval.recall import recall_products
from shopping_assistant.services.events import emit
from shopping_assistant.settings import get_settings

PRODUCTS_BY_ID = {product.product_id: product for product in PRODUCTS}

TOOL_NAMES = (
    "get_shopper_profile",
    "search_catalog",
    "check_inventory",
    "present_recommendation",
)


def product_line(product: Product) -> str:
    """One catalogue row, in the compact form the model reads best."""
    tags = ", ".join(product.tags) if product.tags else "-"
    return (
        f"{product.product_id} | {product.name} | {product.category}"
        f" | {currency_symbol()}{product.price:.2f} | {product.rating}* ({product.rating_count})"
        f" | {product.brand or '-'} | tags: {tags}"
    )


def inventory_summary(items) -> str:
    """One line the UI shows above the product cards."""
    out_of_stock = [item.name for item in items if item.status == "out_of_stock"]
    low_stock = [item.name for item in items if item.status == "low_stock"]
    if out_of_stock:
        return f"Not available right now: {', '.join(out_of_stock)}."
    if low_stock:
        return f"Low stock on {', '.join(low_stock)} — they may sell out soon."
    return "All items are in stock and ready to ship."


# ── tool implementations ──────────────────────────────────────────────


def get_shopper_profile(user_id: str) -> str:
    """Look up the shopper's profile: RFM segment, category preferences, usual budget."""

    profile = build_profile(user_id)
    if not profile:
        return "No profile on file for that shopper."

    emit({"type": "profile", "profile": profile.model_dump(mode="json")})

    symbol = currency_symbol()
    return (
        f"{profile.name} ({profile.tier}), segment {profile.segment}. "
        f"{profile.rfm.orders} orders in 90 days, last purchase {profile.rfm.recency_days} days ago. "
        f"Preferred categories: {', '.join(profile.preferred_categories) or 'unknown'}. "
        f"Usual budget {symbol}{profile.price_range[0]:.0f}-{symbol}{profile.price_range[1]:.0f}."
    )


def search_catalog(
    query: str,
    keywords: list[str] | None = None,
    category: str = "",
    min_price: float | None = None,
    max_price: float | None = None,
    limit: int = 8,
) -> str:
    """Search the catalog for candidates; budget and category are hard filters."""
    settings = get_settings()
    depth = max(1, min(limit, settings.max_candidates))

    search = SearchParams(
        keywords=keywords or [],
        category=category,
        min_price=min_price,
        max_price=max_price,
    )
    products = [p for p in recall_products(search, query=query, limit=depth) if p.stock > 0]

    if not products:
        return "No products matched. Try broader keywords, another category, or a higher budget."

    return "\n".join(product_line(product) for product in products)


def check_inventory(product_ids: list[str]) -> str:
    """Check live stock for specific products."""

    products = [PRODUCTS_BY_ID[pid] for pid in product_ids if pid in PRODUCTS_BY_ID]
    result = check_stock(products)

    emit({
        "type": "inventory",
        "items": [item.model_dump(mode="json") for item in result.items],
        "summary": inventory_summary(result.items),
    })

    lines = [
        f"{item.product_id} | {item.name} | stock {item.stock} | {item.status}"
        + (f" | purchase limit {item.purchase_limit}" if item.purchase_limit else "")
        for item in result.items
    ]
    return "\n".join(lines) or "No matching product ids."


def present_recommendation(product_ids: list[str]) -> str:
    """Show the shopper the product cards for the final picks, best first."""
    settings = get_settings()

    unknown = [pid for pid in product_ids if pid not in PRODUCTS_BY_ID]
    known = [PRODUCTS_BY_ID[pid] for pid in product_ids if pid in PRODUCTS_BY_ID]
    chosen = known[: settings.max_products]

    if not chosen:
        return f"None of those ids exist in the catalog: {', '.join(unknown)}."

    emit({"type": "products", "products": [product.model_dump(mode="json") for product in chosen]})

    # The model must be told exactly what the shopper can see, otherwise it writes
    # about products the cap dropped and its reply stops matching the cards.
    dropped = len(known) - len(chosen)
    note = f" The other {dropped} were dropped: the cards cap at {settings.max_products}." if dropped else ""
    return (
        f"Shown to the shopper: {'; '.join(p.name for p in chosen)}.{note} "
        f"Write your reply about exactly these products and no others."
    )


# ── schemas ───────────────────────────────────────────────────────────

_PROFILE_SCHEMA = {
    "type": "object",
    "properties": {
        "user_id": {"type": "string", "description": "The shopper's id, given to you with every message."}
    },
    "required": ["user_id"],
    "additionalProperties": False,
}

_SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "The shopper's request rewritten as a search phrase, in the catalog's language.",
        },
        "keywords": {
            "type": "array",
            "items": {"type": "string"},
            "description": "1-4 short catalog terms for exact matching (brands, materials, model numbers).",
        },
        "category": {"type": "string", "description": "A catalog category, or \"\" for any category."},
        "min_price": {"type": "number", "description": "Lower bound in the catalog currency."},
        "max_price": {
            "type": "number",
            "description": "Upper bound in the catalog currency. Always pass this when the shopper names a budget.",
        },
        "limit": {"type": "integer", "description": "How many candidates to return, default 8."},
    },
    "required": ["query"],
    "additionalProperties": False,
}

_INVENTORY_SCHEMA = {
    "type": "object",
    "properties": {
        "product_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Product ids from a previous search_catalog result.",
        }
    },
    "required": ["product_ids"],
    "additionalProperties": False,
}

_PRESENT_SCHEMA = {
    "type": "object",
    "properties": {
        "product_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Product ids from a previous search_catalog result, best first.",
        }
    },
    "required": ["product_ids"],
    "additionalProperties": False,
}

DESCRIPTIONS: dict[str, str] = {
    "get_shopper_profile": (
        "Look up the shopper's profile: RFM segment, category preferences and usual budget. "
        "Call this before choosing products when personalisation matters."
    ),
    "search_catalog": (
        "Search the product catalog for candidates to recommend. The budget and category "
        "arguments are hard filters, and sold-out products are never returned."
    ),
    "check_inventory": (
        "Check live stock for specific products. Call this for the products you intend to "
        "recommend, before presenting them."
    ),
    "present_recommendation": (
        "Show the shopper the product cards for your final picks, best first. Call this once, "
        "after checking stock, with the products you actually stand behind. The shopper sees "
        "exactly these cards, so do not include anything you would not defend."
    ),
}


def _invoke(function, arguments: dict[str, Any]) -> str:
    return str(function(**arguments))


def build_tool_registry() -> ToolRegistry:
    """The four serving tools, declared once for the harness."""
    registry = ToolRegistry()
    registry.add(
        "get_shopper_profile",
        DESCRIPTIONS["get_shopper_profile"],
        _PROFILE_SCHEMA,
        lambda arguments: _invoke(get_shopper_profile, arguments),
    )
    registry.add(
        "search_catalog",
        DESCRIPTIONS["search_catalog"],
        _SEARCH_SCHEMA,
        lambda arguments: _invoke(search_catalog, arguments),
    )
    registry.add(
        "check_inventory",
        DESCRIPTIONS["check_inventory"],
        _INVENTORY_SCHEMA,
        lambda arguments: _invoke(check_inventory, arguments),
    )
    registry.add(
        "present_recommendation",
        DESCRIPTIONS["present_recommendation"],
        _PRESENT_SCHEMA,
        lambda arguments: _invoke(present_recommendation, arguments),
    )
    return registry
