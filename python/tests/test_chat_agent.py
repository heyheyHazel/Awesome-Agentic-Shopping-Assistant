"""Tests for ChatAgent — intent parsing and delegation logic."""

from __future__ import annotations

import json as json_module
from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.chat_agent import ChatAgent


# ── Intent parsing tests ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_parse_intent_extracts_category():
    """ChatAgent extracts category from chat messages."""
    response = json_module.dumps({
        "category": "口红",
        "brand": "",
        "price_min": 0,
        "price_max": 500,
        "keyword": "口红",
        "intent_type": "product_search",
        "num_items": 6,
    })

    agent = ChatAgent(supervisor=None)
    msg = MagicMock()
    msg.content = response
    agent.llm = MagicMock(ainvoke=AsyncMock(return_value=msg))

    intent = await agent._parse_intent([
        {"role": "user", "content": "推荐一款口红"}
    ])

    assert intent.intent_type == "product_search"
    assert intent.category == "口红"
    assert intent.keyword == "口红"
    assert intent.price_max == 500


@pytest.mark.asyncio
async def test_parse_intent_extracts_price_range():
    """ChatAgent extracts price range when user specifies budget."""
    response = json_module.dumps({
        "category": "手机",
        "brand": "华为",
        "price_min": 2000,
        "price_max": 5000,
        "keyword": "华为手机",
        "intent_type": "product_search",
        "num_items": 6,
    })

    agent = ChatAgent(supervisor=None)
    msg = MagicMock()
    msg.content = response
    agent.llm = MagicMock(ainvoke=AsyncMock(return_value=msg))

    intent = await agent._parse_intent([
        {"role": "user", "content": "想买一款华为手机，预算2000到5000"}
    ])

    assert intent.intent_type == "product_search"
    assert intent.category == "手机"
    assert intent.brand == "华为"
    assert intent.price_min == 2000
    assert intent.price_max == 5000


@pytest.mark.asyncio
async def test_parse_intent_general_question():
    """ChatAgent detects general (non-shopping) questions."""
    response = json_module.dumps({
        "category": "",
        "brand": "",
        "price_min": 0,
        "price_max": 10000,
        "keyword": "",
        "intent_type": "general_question",
        "num_items": 6,
    })

    agent = ChatAgent(supervisor=None)
    msg = MagicMock()
    msg.content = response
    agent.llm = MagicMock(ainvoke=AsyncMock(return_value=msg))

    intent = await agent._parse_intent([
        {"role": "user", "content": "你好"}
    ])

    assert intent.intent_type == "general_question"
    assert intent.category == ""


@pytest.mark.asyncio
async def test_general_intent_skips_orchestrator():
    """General questions do not call the orchestrator (no supervisor needed)."""
    response = json_module.dumps({
        "category": "",
        "brand": "",
        "price_min": 0,
        "price_max": 10000,
        "keyword": "",
        "intent_type": "general_question",
        "num_items": 6,
    })

    agent = ChatAgent(supervisor=None)
    msg = MagicMock()
    msg.content = response
    agent.llm = MagicMock(ainvoke=AsyncMock(return_value=msg))

    result = await agent.run(
        user_id="U001",
        messages=[{"role": "user", "content": "你好呀"}],
    )

    assert result.success is True
    intent = getattr(result, "intent", None)
    assert intent is not None
    assert intent.intent_type == "general_question"
    assert getattr(result, "products", []) == []


@pytest.mark.asyncio
async def test_fallback_on_llm_error():
    """ChatAgent returns fallback result when LLM raises an exception."""
    agent = ChatAgent(supervisor=None)

    async def _raise(_messages):
        raise RuntimeError("LLM unavailable")

    agent.llm = MagicMock(ainvoke=_raise)

    result = await agent.run(
        user_id="U001",
        messages=[{"role": "user", "content": "推荐口红"}],
    )

    assert result.success is False
    assert "LLM unavailable" in (result.error or "")
    reply = getattr(result, "reply_prompt", "")
    assert "抱歉" in reply or "重试" in reply


@pytest.mark.asyncio
async def test_clean_intent_handles_markdown_wrapper():
    """_clean_intent strips markdown code fences from LLM output."""
    agent = ChatAgent(supervisor=None)
    raw = '```json\n{"category": "耳机", "intent_type": "product_search"}\n```'
    intent = agent._clean_intent(raw)
    assert intent.category == "耳机"
    assert intent.intent_type == "product_search"
