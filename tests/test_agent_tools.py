"""The agent's tools: constraint enforcement, stock filtering, and what reaches the UI.

Tools are deterministic, so everything here runs without an LLM. They emit only
domain payloads — the loop owns the running/done lifecycle of every tool — so the
events captured here are exactly the ones the panels render.
"""

from __future__ import annotations

import pytest

from shopping_assistant.agent import tools


@pytest.fixture
def events(monkeypatch):
    """Capture every payload the tools emit.

    The module-level name is patched rather than the shared sink: a context
    variable set inside a fixture generator does not stay visible to the test
    that resumes it.
    """
    captured: list[dict] = []
    monkeypatch.setattr(tools, "emit", captured.append)
    return captured


def types_of(events: list[dict]) -> list[str]:
    return [event["type"] for event in events]


def registry():
    return tools.build_tool_registry()


def test_the_tool_set_is_exactly_what_the_agent_advertises():
    assert registry().names == list(tools.TOOL_NAMES)


def test_every_tool_is_declared_with_a_schema_the_model_can_call():
    for schema in registry().schemas():
        function = schema["function"]
        assert function["description"]
        assert function["parameters"]["type"] == "object"
        assert function["parameters"]["properties"]


def test_the_registry_never_raises_on_bad_arguments():
    observation, ok = registry().invoke(
        type("Call", (), {"name": "search_catalog", "arguments": {"unexpected": 1}})()
    )
    assert ok is False
    assert "search_catalog failed" in observation


def test_search_honours_the_budget_cap(events):
    rows = tools.search_catalog(query="跑鞋", max_price=700, limit=12)

    prices = [float(line.split("|")[3].strip().lstrip("¥")) for line in rows.splitlines()]
    assert prices
    assert all(price <= 700 for price in prices)


def test_search_never_returns_sold_out_products(events):
    """P005 is the cheapest running shoe in the demo catalog and has no stock."""
    rows = tools.search_catalog(query="running shoes", keywords=["running shoes"], limit=12)

    assert "P005" not in rows


def test_search_respects_the_limit(events):
    rows = tools.search_catalog(query="香薰", limit=3)
    assert len(rows.splitlines()) <= 3


def test_search_reports_an_empty_result_plainly(events):
    rows = tools.search_catalog(query="护肤", max_price=0.5)

    assert "No products matched" in rows
    assert types_of(events) == []


def test_present_recommendation_emits_the_cards(events):
    tools.present_recommendation(product_ids=["P001", "P002"])

    assert types_of(events) == ["products"]
    assert [p["product_id"] for p in events[0]["products"]] == ["P001", "P002"]


def test_present_recommendation_drops_unknown_ids(events):
    result = tools.present_recommendation(product_ids=["P001", "does-not-exist"])

    assert [p["product_id"] for p in events[0]["products"]] == ["P001"]
    assert "Shown to the shopper" in result


def test_present_recommendation_rejects_a_wholly_invented_id(events):
    result = tools.present_recommendation(product_ids=["nope-1", "nope-2"])

    assert events == []
    assert "exist in the catalog" in result


def test_present_recommendation_caps_how_many_cards_are_shown(events):
    from shopping_assistant.settings import get_settings

    tools.present_recommendation(product_ids=list(tools.PRODUCTS_BY_ID))

    assert len(events[0]["products"]) == get_settings().max_products


def test_check_inventory_reports_stock_for_each_product(events):
    result = tools.check_inventory(product_ids=["P001", "P005"])

    assert types_of(events) == ["inventory"]
    statuses = {item["product_id"]: item["status"] for item in events[0]["items"]}
    assert statuses == {"P001": "in_stock", "P005": "out_of_stock"}
    assert "P005" in result


def test_profile_tool_emits_a_dashboard_shaped_payload(events):
    tools.get_shopper_profile(user_id="U001")

    assert types_of(events) == ["profile"]
    profile = events[0]["profile"]
    # The right-hand panel reads exactly these fields.
    assert profile["segment"] == "Champions"
    assert set(profile["rfm"]) >= {"recency", "frequency", "monetary", "overall"}
    assert profile["price_range"] == [400.0, 2000.0]


def test_profile_tool_is_safe_for_an_unknown_shopper(events):
    result = tools.get_shopper_profile(user_id="nobody")

    # `data.get_user` falls back to the demo default rather than raising.
    assert events and events[0]["profile"]["user_id"] == "U001"
    assert result


def test_tools_called_outside_a_bound_turn_emit_nothing():
    """A script or a test that calls a tool directly must not need a transport."""
    tools.present_recommendation(product_ids=["P001"])
