"""Hybrid recall: rank fusion, index validation and the search itself."""

from __future__ import annotations

import numpy as np
import pytest

from shopping_assistant.domain.models import Product, SearchParams
from shopping_assistant.retrieval import index as vector_index
from shopping_assistant.retrieval.recall import recall_products, reciprocal_rank_fusion


def make(product_id: str, name: str = "", price: float = 10.0, **kwargs) -> Product:
    return Product(
        product_id=product_id,
        name=name or f"product {product_id}",
        category=kwargs.pop("category", "Test"),
        price=price,
        **kwargs,
    )


def test_fusion_ranks_agreement_between_retrievers_first():
    keyword = [make("1"), make("2"), make("3")]
    semantic = [make("3"), make("1"), make("4")]

    fused = reciprocal_rank_fusion([keyword, semantic], limit=4)

    # 1 and 3 are found by both retrievers, so they outrank the single-retriever hits.
    assert [p.product_id for p in fused] == ["1", "3", "2", "4"]


def test_fusion_respects_the_limit():
    fused = reciprocal_rank_fusion([[make("1"), make("2"), make("3")]], limit=2)
    assert [p.product_id for p in fused] == ["1", "2"]


def test_hard_filters_still_bound_the_fused_result(monkeypatch):
    """Fusion must not smuggle back a product the constraints excluded."""
    monkeypatch.setattr(
        "shopping_assistant.retrieval.recall.semantic_ranking",
        lambda query, eligible: [make("P999", price=9999)],
    )

    products = recall_products(SearchParams(keywords=["running shoes"], max_price=800), query="cheap shoes")

    assert products
    assert all(p.price <= 800 for p in products)
    assert "P999" not in [p.product_id for p in products]


def test_index_round_trips_and_rejects_a_changed_catalog(tmp_path):
    vectors_file, ids_file = tmp_path / "v.npy", tmp_path / "i.json"
    vectors = np.eye(2, dtype=np.float32)
    vector_index.VectorIndex(["A", "B"], vectors).save(vectors_file, ids_file)

    restored = vector_index.load(["A", "B"], vectors_file=vectors_file, ids_file=ids_file)
    assert restored is not None and restored.ids == ["A", "B"]

    # Same files, different catalog: the index belongs to different rows now.
    assert vector_index.load(["A", "C"], vectors_file=vectors_file, ids_file=ids_file) is None


def test_missing_index_is_not_an_error(tmp_path):
    assert vector_index.load(vectors_file=tmp_path / "nope.npy", ids_file=tmp_path / "nope.json") is None


def test_search_masks_out_products_the_shopper_ruled_out():
    index = vector_index.VectorIndex(
        ["near", "far"], np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    )

    hits = index.search(np.array([1.0, 0.0], dtype=np.float32), allowed_ids={"far"}, top_k=2)

    assert hits == [("far", pytest.approx(0.0))]


def test_search_honours_the_similarity_floor():
    index = vector_index.VectorIndex(["a"], np.array([[1.0, 0.0]], dtype=np.float32))

    assert index.search(np.array([0.2, 0.98], dtype=np.float32), min_similarity=0.9) == []
    assert index.search(np.array([0.2, 0.98], dtype=np.float32), min_similarity=0.1)


def test_recall_respects_budget():
    products = recall_products(SearchParams(keywords=["running shoes"], max_price=700))
    assert products
    assert all(p.price <= 700 for p in products)




def test_recall_matches_keywords():
    products = recall_products(SearchParams(keywords=["coffee"]))
    assert any("Coffee" in p.name for p in products)




def test_recall_falls_back_when_nothing_matches():
    products = recall_products(SearchParams(keywords=["spaceship"]))
    assert len(products) == 12




def test_recall_never_violates_the_hard_filters():
    """An empty result must stay empty rather than fall back to the whole catalog."""
    assert recall_products(SearchParams(keywords=["护肤"], max_price=0.5)) == []
    assert recall_products(SearchParams(keywords=["护肤"], category="不存在的类目")) == []


