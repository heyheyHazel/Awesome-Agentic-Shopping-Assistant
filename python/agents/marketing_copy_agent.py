"""Marketing copy agent: segment-specific product copy via structured LLM output."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from config import get_settings
from models.schemas import CopyResult, CopySet, Product, UserProfile

from .base_agent import BaseAgent
from .models import build_llm
from .structured import JsonStructured

SEGMENT_STYLES = {
    "Champions": "premium and appreciative; emphasise quality and the VIP treatment",
    "Loyal": "warm and familiar; highlight everyday value and new arrivals",
    "Potential": "encouraging; highlight ratings, fit, and why shoppers love it",
    "At Risk": "win-back tone; a light sense of urgency and a reason to return",
    "New": "welcoming; first-order friendly and trust-building",
}

COPY_SYSTEM = """You write short e-commerce product copy.
For each product write ONE sentence of at most 25 words, plain and specific — no hype words like "best ever".
Respond in English. Keep each sentence grounded in the product data (category, price, rating, tags)."""


class MarketingCopyAgent(BaseAgent):
    """Generates one line of personalised copy per recommended product."""

    def __init__(self):
        settings = get_settings()
        super().__init__(name="copy", timeout=settings.agent_timeout_llm)
        llm = build_llm(temperature=0.7, max_tokens=512)
        self.writer = JsonStructured(llm, CopySet)

    async def _execute(self, **kwargs: Any) -> CopyResult:
        """Write copy for each product, styled for the shopper's segment."""
        profile: UserProfile | None = kwargs.get("profile")
        products: list[Product] = kwargs.get("products", [])
        if not products:
            return CopyResult()

        segment = profile.segment if profile else "New"
        style = SEGMENT_STYLES.get(segment, SEGMENT_STYLES["New"])
        product_lines = "\n".join(
            f"- {p.product_id}: {p.name} | {p.category} | ${p.price:.2f} | {p.rating} stars ({p.rating_count} reviews) | tags: {', '.join(p.tags)}"
            for p in products
        )

        messages = [
            SystemMessage(content=f"{COPY_SYSTEM}\nShopper segment: {segment}. Tone: {style}."),
            HumanMessage(content=f"Products:\n{product_lines}"),
        ]
        result = await self.writer.ainvoke(messages)
        return CopyResult(items=result.items)

    def _fallback(self, latency_ms: float, exc: Exception) -> CopyResult:
        return CopyResult(success=False, latency_ms=latency_ms, error=str(exc))
