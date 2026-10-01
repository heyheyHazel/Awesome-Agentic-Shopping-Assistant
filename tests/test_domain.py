"""RFM segmentation, currency and stock rules."""

from shopping_assistant.domain.currency import currency_code, currency_symbol
from shopping_assistant.domain.inventory import check_stock
from shopping_assistant.domain.models import Product
from shopping_assistant.domain.rfm import SEGMENTS, RFMScale, classify, compute_rfm
from shopping_assistant.catalog import USERS, DemoUser


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
    assert rfm.lifetime_value == 8800.00
    # 30% x 0.5 recency + 30% x 1.0 frequency + 40% x 1.0 monetary
    assert rfm.overall == 0.85




def test_every_segment_stays_populated_on_a_large_shopper_base():
    """Percentile segments must not collapse into one bucket as the base grows."""
    base = [
        DemoUser(
            user_id=f"X{index}", name="", email="", tier="Standard",
            recency_days=1 + index % 88,
            orders=1 + (index * 7) % 60,
            lifetime_value=100.0 + (index * 137) % 8000,
        )
        for index in range(200)
    ]
    scale = RFMScale(base)
    segments = {classify(user, scale) for user in base}
    assert segments == set(SEGMENTS)




def test_currency_defaults_to_cny():
    assert currency_code() == "CNY"
    assert currency_symbol() == "¥"




def test_inventory_classifies_stock_levels():
    products = [
        Product(product_id="X", name="Sold Out", category="c", price=1, stock=0),
        Product(product_id="Y", name="Almost Gone", category="c", price=1, stock=5),
        Product(product_id="Z", name="Plenty", category="c", price=1, stock=100),
    ]
    result = check_stock(products)

    statuses = {item.product_id: item.status for item in result.items}
    assert statuses == {"X": "out_of_stock", "Y": "low_stock", "Z": "in_stock"}
    assert result.available_ids == ["Y", "Z"]
    assert result.items[1].purchase_limit == 1
