"""Chat agent: a tool-calling assistant for general questions."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool

from agents.user_profile_agent import classify, compute_rfm
from data.products import PRODUCTS
from data.users import get_user

from .language import language_directive
from .models import build_llm

CHAT_SYSTEM = """You are a friendly shopping assistant for an online store.
Use the tools to look up the catalog and the shopper's profile before answering.
Stay concrete and helpful, and keep the reply under three sentences."""


@tool
def search_catalog(query: str, max_price: float | None = None) -> str:
    """Search the product catalog by keywords and optional budget cap.

    Args:
        query: Lowercase keywords, e.g. "running shoes" or "skincare".
        max_price: Optional maximum price in USD.
    """
    terms = [term for term in query.lower().split() if term]
    matches: list[str] = []
    for product in PRODUCTS:
        if max_price is not None and product.price > max_price:
            continue
        haystack = f"{product.name} {product.category} {product.brand} {' '.join(product.tags)}".lower()
        if terms and not any(term in haystack for term in terms):
            continue
        matches.append(
            f"{product.product_id} | {product.name} | {product.category} | ${product.price:.2f} "
            f"| {product.rating} stars ({product.rating_count}) | stock {product.stock}"
        )
    return "\n".join(matches[:6]) if matches else "No matching products found."


@tool
def get_shopper_profile(user_id: str) -> str:
    """Return a shopper's RFM segment, preferences and budget by user id.

    Args:
        user_id: The shopper's id, e.g. "U001".
    """
    user = get_user(user_id)
    rfm = compute_rfm(user)
    return (
        f"{user.name} ({user.tier}), segment {classify(user)}, {rfm.orders} orders, "
        f"last purchase {rfm.recency_days} days ago, lifetime value ${rfm.lifetime_value:.2f}, "
        f"prefers {', '.join(user.preferred_categories) or 'unknown'}, "
        f"budget ${user.price_range[0]:.0f}-${user.price_range[1]:.0f}"
    )


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return ""


class ChatAgent:
    """Wraps a LangChain tool-calling agent and streams its reply tokens."""

    def __init__(self):
        model = build_llm(temperature=0.6, max_tokens=512)
        self.agent = create_agent(
            model=model,
            tools=[search_catalog, get_shopper_profile],
            system_prompt=CHAT_SYSTEM,
        )

    async def astream(self, messages: list[Any], user_id: str, language: str = "en") -> AsyncIterator[str]:
        """Yield reply text chunks for the conversation so far."""
        inputs = {
            "messages": [
                SystemMessage(
                    content=(
                        f"The current shopper's user_id is {user_id}. "
                        f"{language_directive(language)}"
                    )
                ),
                *messages,
            ]
        }
        async for chunk, _meta in self.agent.astream(inputs, stream_mode="messages"):
            text = _text(getattr(chunk, "content", ""))
            if text:
                yield text
