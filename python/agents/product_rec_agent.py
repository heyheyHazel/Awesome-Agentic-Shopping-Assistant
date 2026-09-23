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
from .language import language_directive
from .models import build_llm
from .structured import JsonStructured
from services.embeddings import get_encoder, model_is_present
from services.vector_index import get_index, product_text

RERANK_SYSTEM = """You are the recommendation model of an e-commerce shopping assistant.
Do both of these jobs in one reply.

1. Rank the candidate products so the best match for the shopper's request comes first.
   Criteria, in order of importance:
   - Relevance to the request
   - Price within the shopper's budget
   - Rating quality and review volume
   - Category diversity across the top results

2. Write a pitch line for each ranked product: one sentence of at most 25 words, plain and
   specific, grounded in the product data (category, price, rating, tags). No hype words like
   "best ever". When a shopper segment is given, match its tone.

Rank only IDs from the candidate list, and give every ranked product exactly one pitch.
If none of the candidates genuinely fit the request, return an empty product list — saying
there is nothing suitable beats padding the answer with unrelated products."""


RRF_K = 60  # standard Reciprocal Rank Fusion damping constant


def _matches_filters(product: Product, search: SearchParams) -> bool:
    """Hard constraints the shopper stated: category, brand and price bounds."""
    category = search.category.lower().strip()
    brand = search.brand.lower().strip()
    if category and category not in product.category.lower():
        return False
    if brand and brand not in product.brand.lower():
        return False
    if search.min_price is not None and product.price < search.min_price:
        return False
    if search.max_price is not None and product.price > search.max_price:
        return False
    return True


# Lowercased search text per product, built once. Rebuilding the string for every
# product on every query was the bulk of recall time at a 23k-product catalog.
_haystacks: dict[str, str] = {}


def _haystack(product: Product) -> str:
    cached = _haystacks.get(product.product_id)
    if cached is None:
        cached = product_text(product).lower()
        _haystacks[product.product_id] = cached
    return cached


def _keyword_ranking(products: list[Product], terms: list[str]) -> list[Product]:
    """Rank by keyword hits first, then by rating quality.

    A product with no hit still scores its rating, so this alone degenerates into
    "top rated" as soon as nothing matches; the semantic ranking is what keeps the
    result about the request rather than about review counts.
    """
    def score(product: Product) -> float:
        haystack = _haystack(product)
        return sum(1 for term in terms if term in haystack) * 2.0 + product.rating

    return sorted(products, key=score, reverse=True)


def _semantic_ranking(query: str, eligible: list[Product]) -> list[Product]:
    """Nearest catalog products by embedding similarity, or [] when unavailable."""
    if not query.strip() or not model_is_present():
        return []
    index = get_index()
    if index is None:
        return []

    by_id = {product.product_id: product for product in eligible}
    vector = get_encoder().encode([query])[0]
    hits = index.search(vector, allowed_ids=set(by_id))
    return [by_id[pid] for pid, _ in hits if pid in by_id]


def _reciprocal_rank_fusion(rankings: list[list[Product]], limit: int) -> list[Product]:
    """Merge rankings by rank position, so agreement between retrievers wins.

    Fusing ranks instead of scores avoids having to calibrate a keyword score
    against a cosine similarity.
    """
    scores: dict[str, float] = {}
    by_id: dict[str, Product] = {}
    for ranking in rankings:
        for rank, product in enumerate(ranking, start=1):
            by_id[product.product_id] = product
            scores[product.product_id] = scores.get(product.product_id, 0.0) + 1.0 / (RRF_K + rank)
    ordered = sorted(scores.items(), key=lambda item: -item[1])
    return [by_id[pid] for pid, _ in ordered[:limit]]


def recall_products(search: SearchParams, query: str = "", limit: int = 12) -> list[Product]:
    """Shortlist the catalog for one turn: hard filters, then keyword + semantic ranking.

    When the constraints rule everything out the result is empty: a shopper who
    asked for something under a budget must not be shown items over it, so the
    constraints are never traded away to keep the list populated.
    """
    terms = [t.lower().strip() for t in search.keywords if t.strip()]
    eligible = [product for product in PRODUCTS if _matches_filters(product, search)]
    if not eligible:
        return []

    keyword = _keyword_ranking(eligible, terms)
    semantic = _semantic_ranking(query, eligible)
    if not semantic:
        shortlist = keyword[:limit]
    else:
        shortlist = _reciprocal_rank_fusion([keyword, semantic], limit)

    # Enforce the constraints at the exit rather than trusting every retriever to
    # have applied them, so the guarantee holds however a retriever is changed.
    return [product for product in shortlist if _matches_filters(product, search)]


class ProductRecAgent(BaseAgent):
    """Ranks the recall candidates and writes their pitch lines in one LLM call."""

    def __init__(self):
        settings = get_settings()
        super().__init__(name="rerank", timeout=settings.agent_timeout_llm)
        # Ranking and copy share one reply, so the budget covers both.
        llm = build_llm(temperature=0.2, max_tokens=900)
        self.ranker = JsonStructured(llm, ProductRanking)

    async def _execute(self, **kwargs: Any) -> ProductRecResult:
        """Rank the candidates and write one pitch line for each."""
        profile: UserProfile | None = kwargs.get("profile")
        products: list[Product] = kwargs.get("products", [])
        query: str = kwargs.get("query", "")
        language: str = kwargs.get("language", "en")
        k: int = kwargs.get("k", 3)

        if not products:
            return ProductRecResult()

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
            SystemMessage(content=f"{RERANK_SYSTEM}\n{language_directive(language)}"),
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
        # Pad only a *partial* ranking. An empty product_ids list is the model saying
        # nothing fits, and must not be turned back into the shortlist.
        if ranked:
            seen = {p.product_id for p in ranked}
            ranked.extend(p for p in products if p.product_id not in seen)
        ranked = ranked[:k]

        order = {p.product_id: index for index, p in enumerate(ranked)}
        pitches = sorted(
            (pitch for pitch in ranking.pitches if pitch.product_id in order),
            key=lambda pitch: order[pitch.product_id],
        )
        return ProductRecResult(products=ranked, pitches=pitches)

    def _fallback(self, latency_ms: float, exc: Exception) -> ProductRecResult:
        return ProductRecResult(success=False, latency_ms=latency_ms, error=str(exc))
