"""Agent behaviour: RFM profiling, catalog recall, inventory classification."""

from agents.inventory_agent import InventoryAgent
from agents.product_rec_agent import recall_products
from agents.user_profile_agent import classify, compute_rfm
from data.users import USERS
from models.schemas import Product, SearchParams


def test_each_demo_user_gets_a_distinct_segment():
    expected = {"U001": "Champions", "U002": "New", "U003": "Loyal", "U004": "At Risk"}
    for user_id, segment in expected.items():
        user = USERS[user_id]
        assert classify(user) == segment
        rfm = compute_rfm(user)
        assert 0.0 <= rfm.overall <= 1.0


def test_emily_rfm_matches_dashboard_values():
    rfm = compute_rfm(USERS["U001"])
    assert rfm.recency_days == 5
    assert rfm.orders == 12
    assert rfm.lifetime_value == 1256.00
    assert rfm.overall == 0.918


def test_recall_respects_budget():
    products = recall_products(SearchParams(keywords=["running shoes"], max_price=100))
    assert products
    assert all(p.price <= 100 for p in products)


def test_recall_matches_keywords():
    products = recall_products(SearchParams(keywords=["coffee"]))
    assert any("Coffee" in p.name for p in products)


def test_recall_falls_back_when_nothing_matches():
    products = recall_products(SearchParams(keywords=["spaceship"]))
    assert len(products) == 12


async def test_inventory_classifies_stock_levels():
    products = [
        Product(product_id="X", name="Sold Out", category="c", price=1, stock=0),
        Product(product_id="Y", name="Almost Gone", category="c", price=1, stock=5),
        Product(product_id="Z", name="Plenty", category="c", price=1, stock=100),
    ]
    result = await InventoryAgent().run(products=products)

    statuses = {item.product_id: item.status for item in result.items}
    assert statuses == {"X": "out_of_stock", "Y": "low_stock", "Z": "in_stock"}
    assert result.available_ids == ["Y", "Z"]
    assert result.items[1].purchase_limit == 1
