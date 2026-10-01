"""Verifiable reward for one purchase, ported from ShopSimulator.

Four sub-scores are produced and kept separate because they measure different
failures: buying the wrong kind of product, missing a requested attribute,
missing a requested option, or overshooting the price ceiling. The scalar
``r_loose`` is the upstream weighted count; ``r_hard`` (the product of the four)
is what "strictly solved the task" means.

Divergences from upstream, all deliberate:

* ``r_type`` keeps upstream's query/category/title test, but the noun-overlap
  term uses CJK character bigrams instead of spaCy ``zh_core_web_sm``. The
  shipped records carry no ``query`` field, so upstream's query term compares
  "" with "" and already saturates to 1.0 on this data; the title term is the
  only part that can still move.
* Fuzzy matching uses ``rapidfuzz`` when installed and ``difflib`` otherwise.
  Both are applied at the same 85 score threshold as upstream.
* Prices and price ceilings are deterministic functions of the ASIN and the
  instruction instead of process-level random state, so the same task scores
  the same on every rollout.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field

from shoprl.env.search import tokenize

SUB_SCORES = ("r_type", "r_att", "r_option", "r_price")
FUZZY_THRESHOLD = 85.0
NO_PRICE_CEILING = 10_000_000.0


@dataclass(slots=True)
class Goal:
    """What the task asks for, derived from the target product's instruction."""

    asin: str
    title: str
    category: str
    instruction: str
    attributes: list[str] = field(default_factory=list)
    options: list[str] = field(default_factory=list)
    price_upper: float = NO_PRICE_CEILING
    query: str = ""


@dataclass(slots=True)
class Purchase:
    """What the agent bought."""

    asin: str
    title: str
    category: str
    price: float
    attributes: list[str] = field(default_factory=list)
    options: list[str] = field(default_factory=list)
    query: str = ""


def _ratio(left: str, right: str) -> float:
    try:
        from rapidfuzz.fuzz import token_set_ratio

        return float(token_set_ratio(left, right))
    except ImportError:
        from difflib import SequenceMatcher

        return SequenceMatcher(None, left, right).ratio() * 100


def normalize_option(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def price_ceiling(price: float) -> list[float]:
    """Upstream's candidate ceiling ladder, reproduced as-is (``count`` is ignored)."""
    if price <= 100:
        count = 3
    elif price <= 1000:
        count = 10
    elif price <= 5000:
        count = 50
    elif price <= 10000:
        count = 100
    else:
        count = 100
    base = math.ceil(price / 10) * 10
    return [base + index * 10 for index in range(count)]


def _seed(*parts: str) -> int:
    return int(hashlib.sha256(":".join(parts).encode("utf-8")).hexdigest()[:12], 16)


def deterministic_price(asin: str, pricing: list[float]) -> float:
    """Stable stand-in for upstream's ``random.uniform`` price draw."""
    if not pricing:
        return 100.0
    if len(pricing) == 1:
        return float(pricing[0])
    low, high = float(pricing[0]), float(pricing[1])
    if high <= low:
        return low
    return round(random.Random(_seed("price", asin)).uniform(low, high), 2)


def deterministic_price_upper(asin: str, instruction: str, price: float) -> float:
    """Stable stand-in for upstream's ``random.sample`` price ceiling."""
    ladder = price_ceiling(price)
    if len(ladder) < 2:
        return NO_PRICE_CEILING
    _, high = sorted(random.Random(_seed("ceiling", asin, instruction)).sample(ladder, 2))
    return high


def _type_reward(purchase: Purchase, goal: Goal) -> tuple[float, dict[str, float]]:
    query_match = purchase.query == goal.query
    purchase_parts = {part.strip() for part in purchase.category.split("›") if part.strip()}
    goal_parts = {part.strip() for part in goal.category.split("›") if part.strip()}
    category_match = len(purchase_parts & goal_parts) >= 2

    purchase_tokens = {token for token in tokenize(purchase.title) if len(token) > 1}
    goal_tokens = {token for token in tokenize(goal.title) if len(token) > 1}
    title_score = (
        len(purchase_tokens & goal_tokens) / len(goal_tokens) if goal_tokens else 0.2
    )
    matched = query_match or category_match or title_score > 0.2
    return (
        (1.0 if matched else 0.5),
        {
            "query_match": float(query_match),
            "category_match": float(category_match),
            "title_score": round(title_score, 4),
        },
    )


def _attribute_reward(purchase: Purchase, goal: Goal) -> tuple[float, int]:
    if not goal.attributes:
        return 1.0, 0
    haystack = purchase.attributes + [purchase.title]
    matches = sum(
        1
        for wanted in goal.attributes
        if any(_ratio(candidate, wanted) > FUZZY_THRESHOLD for candidate in haystack)
    )
    return matches / len(goal.attributes), matches


def _option_reward(purchase: Purchase, goal: Goal) -> tuple[float, int]:
    if not goal.options:
        return 1.0, 0
    bought = [normalize_option(value) for value in purchase.options]
    matches = sum(
        1
        for wanted in goal.options
        if any(_ratio(candidate, normalize_option(wanted)) > FUZZY_THRESHOLD for candidate in bought)
    )
    return matches / len(goal.options), matches


def score_purchase(purchase: Purchase, goal: Goal) -> tuple[float, dict[str, float]]:
    """Return ``(r_loose, detail)`` where detail holds the four sub-scores."""
    type_score, type_detail = _type_reward(purchase, goal)
    attribute_score, attribute_matches = _attribute_reward(purchase, goal)
    option_score, option_matches = _option_reward(purchase, goal)
    price_ok = 1.0 if goal.price_upper <= 0 else float(purchase.price <= goal.price_upper)

    denominator = len(goal.attributes) + len(goal.options) + 1
    loose = (attribute_matches + option_matches + price_ok) / denominator * type_score
    detail = {
        "r_type": type_score,
        "r_att": round(attribute_score, 6),
        "r_option": round(option_score, 6),
        "r_price": price_ok,
        "num_attr_matches": float(attribute_matches),
        "num_option_matches": float(option_matches),
        **type_detail,
    }
    detail["r_hard"] = round(
        detail["r_type"] * detail["r_att"] * detail["r_option"] * detail["r_price"], 6
    )
    return round(loose, 6), detail
