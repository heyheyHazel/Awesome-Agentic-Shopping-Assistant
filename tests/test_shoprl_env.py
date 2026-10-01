"""Environment behaviour: catalogue integrity, reward semantics, session flow."""

from __future__ import annotations

import pytest

from shoprl.env.catalog import Catalog, Product, ProductOption
from shoprl.env.local import EnvPool
from shoprl.env.reward import (
    Goal,
    Purchase,
    deterministic_price,
    deterministic_price_upper,
    score_purchase,
)
from shoprl.env.search import Bm25Index, tokenize
from shoprl.env.session import ShopSession
from shoprl.harness.backends import ScriptedBackend
from shoprl.harness.rollout import EpisodeRunner
from shoprl.harness.types import ModelResponse, ToolCall


def make_product(index: int, asin: str, title: str, **kwargs) -> Product:
    options = kwargs.pop(
        "options",
        {"颜色分类": [ProductOption("黑色", 120.0), ProductOption("白色", 150.0)]},
    )
    return Product(
        index=index,
        asin=asin,
        title=title,
        shop_name=kwargs.pop("shop_name", "店铺"),
        category=kwargs.pop("category", "服饰›上衣›卫衣"),
        domain="服饰",
        attributes=kwargs.pop("attributes", ["纯棉", "宽松"]),
        pricing=kwargs.pop("pricing", [120.0, 150.0]),
        options=options,
        instruction=kwargs.pop("instruction", "帮我买一件纯棉宽松的黑色卫衣"),
        instruction_attributes=kwargs.pop("instruction_attributes", ["纯棉", "宽松"]),
        instruction_options=kwargs.pop("instruction_options", ["黑色"]),
        **kwargs,
    )


@pytest.fixture
def catalog() -> Catalog:
    products = [
        make_product(0, "A0001", "纯棉宽松卫衣男黑色", pricing=[120.0]),
        make_product(1, "A0002", "薄款运动裤女夏季速干", pricing=[88.0], category="服饰›裤子›运动裤"),
        make_product(2, "A0003", "保温杯不锈钢大容量", pricing=[59.0], category="家居›水具›保温杯"),
    ]
    return Catalog(products)


@pytest.fixture
def index(catalog: Catalog) -> Bm25Index:
    return Bm25Index.build(catalog)


def test_catalogue_preserves_the_task_index_space(catalog: Catalog):
    assert [product.index for product in catalog] == [0, 1, 2]
    assert catalog.at(1).asin == "A0002"
    assert catalog.get("A0003").title.startswith("保温杯")


def test_tokenizer_emits_cjk_bigrams_and_latin_words():
    tokens = tokenize("黑色 Hoodie 2024")
    assert "hoodie" in tokens and "2024" in tokens
    assert "黑色" in tokens
    # Single characters are not indexed: they match almost any Chinese title.
    assert "黑" not in tokens


def test_search_is_deterministic_and_ranks_the_match_first(catalog: Catalog, index: Bm25Index):
    first = index.search(["保温杯"])
    assert first and first[0] == 2
    assert first == index.search(["保温杯"])
    assert index.search(["不存在的词"]) == []


def test_prices_and_ceilings_are_stable_across_calls():
    assert deterministic_price("A1", [10.0, 20.0]) == deterministic_price("A1", [10.0, 20.0])
    assert deterministic_price("A1", [10.0]) == 10.0
    ceiling = deterministic_price_upper("A1", "给我一个十块钱的东西", 100.0)
    assert ceiling == deterministic_price_upper("A1", "给我一个十块钱的东西", 100.0)
    assert ceiling > 100.0


def test_a_perfect_purchase_scores_one():
    goal = Goal(
        asin="A1",
        title="纯棉宽松卫衣",
        category="服饰›上衣›卫衣",
        instruction="x",
        attributes=["纯棉", "宽松"],
        options=["黑色"],
        price_upper=200.0,
    )
    purchase = Purchase(
        asin="A1",
        title="纯棉宽松卫衣",
        category="服饰›上衣›卫衣",
        price=120.0,
        attributes=["纯棉", "宽松"],
        options=["黑色"],
    )
    loose, detail = score_purchase(purchase, goal)
    assert loose == 1.0
    assert detail["r_hard"] == 1.0
    assert all(detail[key] == 1.0 for key in ("r_type", "r_att", "r_option", "r_price"))


def test_each_missed_requirement_shows_up_in_its_own_sub_score():
    goal = Goal(
        asin="A1",
        title="纯棉宽松卫衣",
        category="服饰›上衣›卫衣",
        instruction="x",
        attributes=["纯棉", "宽松"],
        options=["黑色"],
        price_upper=100.0,
    )
    purchase = Purchase(
        asin="A1",
        title="纯棉宽松卫衣",
        category="服饰›上衣›卫衣",
        price=120.0,
        attributes=["纯棉"],
        options=[],
    )
    loose, detail = score_purchase(purchase, goal)
    assert detail["r_att"] == 0.5
    assert detail["r_option"] == 0.0
    assert detail["r_price"] == 0.0
    assert detail["r_hard"] == 0.0
    # (1 attribute + 0 options + 0 price) / (2 + 1 + 1) * type
    assert loose == pytest.approx(0.25)


def test_session_walks_search_to_purchase(catalog: Catalog, index: Bm25Index):
    session = ShopSession(catalog, index, catalog.at(0))
    session.reset()
    assert "shop" in session.clickables() or "search" in session.clickables()

    session.step("search[纯棉 卫衣]")
    assert catalog.at(0).asin.lower() in session.clickables()

    session.step("click[A0001]")
    assert "buy now" in session.clickables()

    session.step("click[黑色]")
    assert session.options == {"颜色分类": "黑色"}

    outcome = session.step("click[buy now]")
    assert outcome.done and outcome.reward == 1.0
    assert outcome.purchase["asin"] == "A0001"
    assert outcome.goal["asin"] == "A0001"


def test_an_unavailable_action_changes_nothing(catalog: Catalog, index: Bm25Index):
    session = ShopSession(catalog, index, catalog.at(0))
    session.reset()
    before = session.observation()
    outcome = session.step("click[buy now]")
    assert outcome.changed is False
    assert outcome.done is False
    assert session.observation() == before


def test_buying_the_wrong_option_loses_the_option_score(catalog: Catalog, index: Bm25Index):
    session = ShopSession(catalog, index, catalog.at(0))
    session.reset()
    session.step("search[卫衣]")
    session.step("click[A0001]")
    session.step("click[白色]")
    outcome = session.step("click[buy now]")
    assert outcome.reward_detail["r_option"] == 0.0
    assert outcome.reward_detail["r_hard"] == 0.0


def test_a_finished_episode_returns_its_environment_to_the_pool(catalog: Catalog, index: Bm25Index):
    """A leaked slot is invisible until the pool silently runs out."""
    pool = EnvPool(catalog, index, capacity=1)
    step = lambda value: ModelResponse(
        tool_calls=[ToolCall(id=value, name="shop_act", arguments={"action": value})]
    )
    backend = ScriptedBackend(
        [
            ModelResponse(tool_calls=[ToolCall(id="r", name="shop_reset", arguments={})]),
            step("search[卫衣]"),
            step("click[A0001]"),
            step("click[黑色]"),
            step("click[buy now]"),
        ],
        repeat_last=False,
    )

    trajectory = EpisodeRunner(backend, pool, max_turns=10).run(0)

    assert trajectory.termination == "done" and trajectory.reward == 1.0
    assert pool.status()["free"] == 1
    # The slot is genuinely reusable, not just marked free.
    assert EpisodeRunner(backend, pool, max_turns=10).run(0).termination in {"done", "turn_limit"}
