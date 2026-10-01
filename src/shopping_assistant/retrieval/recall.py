"""Catalog recall: hard filters, keyword and semantic retrieval, rank fusion.

Pure functions over the loaded catalog — no LLM, no network, no per-request state —
so the search tool can call this directly and it is cheap to test.

Ranking is a two-stage retrieval problem: lexical matching is precise for brands,
model numbers and materials, embeddings handle the shopper's own wording, and
Reciprocal Rank Fusion combines them without needing to calibrate a keyword score
against a cosine similarity.
"""

from __future__ import annotations

from shopping_assistant.catalog import PRODUCTS
from shopping_assistant.domain.models import Product, SearchParams
from shopping_assistant.retrieval.embeddings import get_encoder, model_is_present
from shopping_assistant.retrieval.index import get_index, product_text
from shopping_assistant.settings import get_settings

RRF_K = 60  # standard Reciprocal Rank Fusion damping constant

# Lowercased search text per product, built once. Rebuilding the string for every
# product on every query was the bulk of recall time at a 23k-product catalog.
_haystacks: dict[str, str] = {}


def _haystack(product: Product) -> str:
    cached = _haystacks.get(product.product_id)
    if cached is None:
        cached = product_text(product).lower()
        _haystacks[product.product_id] = cached
    return cached


def matches_filters(product: Product, search: SearchParams) -> bool:
    """Hard constraints the shopper stated: category, brand and price bounds."""
    category = search.category.lower().strip()
    brand = search.brand.lower().strip()
    if category and category not in product.category.lower():
        return False
    if brand and brand not in product.brand.lower():
        return False
    if search.min_price is not None and product.price < search.min_price:
        return False
    return search.max_price is None or product.price <= search.max_price


def keyword_ranking(products: list[Product], terms: list[str]) -> list[Product]:
    """Rank by keyword hits first, then by rating quality.

    A product with no hit still scores its rating, so this alone degenerates into
    "top rated" as soon as nothing matches; the semantic ranking is what keeps the
    result about the request rather than about review counts.
    """

    def score(product: Product) -> float:
        haystack = _haystack(product)
        return sum(1 for term in terms if term in haystack) * 2.0 + product.rating

    return sorted(products, key=score, reverse=True)


def semantic_ranking(query: str, eligible: list[Product]) -> list[Product]:
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


def reciprocal_rank_fusion(rankings: list[list[Product]], limit: int) -> list[Product]:
    """Merge rankings by rank position, so agreement between retrievers wins."""
    scores: dict[str, float] = {}
    by_id: dict[str, Product] = {}
    for ranking in rankings:
        for rank, product in enumerate(ranking, start=1):
            by_id[product.product_id] = product
            scores[product.product_id] = scores.get(product.product_id, 0.0) + 1.0 / (RRF_K + rank)
    ordered = sorted(scores.items(), key=lambda item: -item[1])
    return [by_id[pid] for pid, _ in ordered[:limit]]


def recall_products(
    search: SearchParams, query: str = "", limit: int | None = None
) -> list[Product]:
    """Shortlist the catalog for one turn: hard filters, then keyword + semantic ranking.

    When the constraints rule everything out the result is empty. A shopper who asked
    for something under a budget must not be shown items over it, so the constraints
    are never traded away to keep the list populated.
    """
    depth = limit if limit is not None else get_settings().max_candidates
    terms = [t.lower().strip() for t in search.keywords if t.strip()]
    eligible = [product for product in PRODUCTS if matches_filters(product, search)]
    if not eligible:
        return []

    keyword = keyword_ranking(eligible, terms)
    semantic = semantic_ranking(query, eligible)
    shortlist = keyword[:depth] if not semantic else reciprocal_rank_fusion([keyword, semantic], depth)

    # Enforce the constraints at the exit rather than trusting every retriever to
    # have applied them, so the guarantee holds however a retriever is changed.
    return [product for product in shortlist if matches_filters(product, search)]
