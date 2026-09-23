"""Product recommendation agent: deterministic catalog recall plus LLM re-ranking."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from config import get_settings
from config.currency import currency_symbol
from data import PRODUCTS
from models.schemas import (
    Product,
    ProductRanking,
    ProductRecResult,
    SearchParams,
    UserProfile,
)

from .base_agent import BaseAgent
from .models import build_llm
from .structured import JsonStructured

RERANK_SYSTEM = """You are the ranking model of an e-commerce recommendation system.
Order the candidate products so the best match for the shopper's request comes first.

Ranking criteria, in order of importance:
1. Relevance to the search terms and category
2. Price within the shopper's budget
3. Rating quality and review volume
4. Category diversity across the top results

Return the product IDs from best to worst match. Only use IDs from the candidate list."""


def recall_products(search: SearchParams, limit: int = 12) -> list[Product]:
    """Filter the catalog by the supervisor's search params, ranked by keyword hits and rating."""
    terms = [t.lower().strip() for t in search.keywords if t.strip()]
    category = search.category.lower().strip()
    brand = search.brand.lower().strip()

    scored: list[tuple[float, Product]] = []
    for product in PRODUCTS:
        haystack = f"{product.name} {product.category} {product.brand} {' '.join(product.tags)}".lower()
        hits = sum(1 for term in terms if term in haystack)

        if category and category not in product.category.lower():
            continue
        if brand and brand not in product.brand.lower():
            continue
        if search.min_price is not None and product.price < search.min_price:
            continue
        if search.max_price is not None and product.price > search.max_price:
            continue

        scored.append((hits * 2.0 + product.rating, product))

    if not scored:
        scored = [(product.rating, product) for product in PRODUCTS]

    scored.sort(key=lambda item: item[0], reverse=True)
    return [product for _, product in scored[:limit]]


class ProductRecAgent(BaseAgent):
    """Recalls products from the catalog and re-ranks them with the LLM."""

    def __init__(self):
        settings = get_settings()
        super().__init__(name="rerank", timeout=settings.agent_timeout_llm)
        llm = build_llm(temperature=0.2, max_tokens=512)
        self.ranker = JsonStructured(llm, ProductRanking)

    async def _execute(self, **kwargs: Any) -> ProductRecResult:
        """Ask the LLM to re-order the recall candidates for this shopper."""
        profile: UserProfile | None = kwargs.get("profile")
        products: list[Product] = kwargs.get("products", [])
        query: str = kwargs.get("query", "")
        k: int = kwargs.get("k", 3)

        if len(products) <= k:
            return ProductRecResult(products=products[:k])

        profile_line = "No shopper profile available."
        if profile:
            symbol = currency_symbol()
            profile_line = (
                f"Segment: {profile.segment} (RFM {profile.rfm.overall}). "
                f"Preferred categories: {', '.join(profile.preferred_categories) or 'unknown'}. "
                f"Budget: {symbol}{profile.price_range[0]:.0f}-{symbol}{profile.price_range[1]:.0f}."
            )

        candidates = [
            {
                "id": p.product_id,
                "name": p.name,
                "category": p.category,
                "price": p.price,
                "rating": p.rating,
                "tags": p.tags,
            }
            for p in products
        ]

        messages = [
            SystemMessage(content=RERANK_SYSTEM),
            HumanMessage(
                content=(
                    f"Shopper request: {query}\n{profile_line}\n\n"
                    f"Candidates: {candidates}"
                )
            ),
        ]
        ranking = await self.ranker.ainvoke(messages)

        by_id = {p.product_id: p for p in products}
        ranked = [by_id[pid] for pid in ranking.product_ids if pid in by_id]
        seen = {p.product_id for p in ranked}
        ranked.extend(p for p in products if p.product_id not in seen)
        return ProductRecResult(products=ranked[:k])

    def _fallback(self, latency_ms: float, exc: Exception) -> ProductRecResult:
        return ProductRecResult(success=False, latency_ms=latency_ms, error=str(exc))
