"""Supervisor agent: routes each turn and produces the execution plan."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from config import get_settings
from data.products import PRODUCTS
from models.schemas import SupervisorPlan, SupervisorResult, UserProfile

from .base_agent import BaseAgent
from .models import build_llm
from .structured import JsonStructured

CATEGORIES = sorted({product.category for product in PRODUCTS})

SUPERVISOR_SYSTEM = f"""You are the supervisor of a multi-agent shopping assistant.
Decide how to handle the shopper's latest message.

- intent: "product_search" if they want product recommendations or ask about products, prices or stock; else "general".
- reply: one short sentence shown immediately, e.g. "Got it! Here are the best running shoes for you."
- search: catalog filters. Use the previous conversation to resolve follow-ups like "cheaper ones".
  keywords: 1-3 short lowercase terms. category: one of [{', '.join(CATEGORIES)}] when it clearly matches, else "".
  Fill max_price/min_price only when the shopper states a budget.
- agents: pipeline stages to run this turn.
  For product_search always include "recall", "rerank", "inventory";
  include "profile" when personalisation matters (most cases);
  include "copy" when marketing copy adds value (recommendations), skip it for quick factual checks.
  For general: leave the list empty.
"""


class SupervisorAgent(BaseAgent):
    """LLM-based planner: the dynamic scheduler of the agent pipeline."""

    def __init__(self):
        settings = get_settings()
        super().__init__(name="supervisor", timeout=settings.agent_timeout_llm)
        llm = build_llm(temperature=0.1, max_tokens=512)
        self.planner = JsonStructured(llm, SupervisorPlan)

    async def _execute(
        self,
        messages: list[Any] | None = None,
        profile: UserProfile | None = None,
        **_: Any,
    ) -> SupervisorResult:
        """Ask the LLM for the plan of this turn."""
        context = ""
        if profile:
            context = (
                f"\nShopper: {profile.name}, segment {profile.segment}, "
                f"prefers {', '.join(profile.preferred_categories)}, "
                f"budget ${profile.price_range[0]:.0f}-${profile.price_range[1]:.0f}."
            )
        prompt = [SystemMessage(content=SUPERVISOR_SYSTEM + context), *(messages or [])]
        plan = await self.planner.ainvoke(prompt)
        return SupervisorResult(plan=plan)

    def _fallback(self, latency_ms: float, exc: Exception) -> SupervisorResult:
        """Fall back to a general answer so the conversation never breaks."""
        plan = SupervisorPlan(
            intent="general",
            reply="Sorry, I had trouble understanding that — could you rephrase?",
            agents=[],
        )
        return SupervisorResult(success=False, latency_ms=latency_ms, error=str(exc), plan=plan)
