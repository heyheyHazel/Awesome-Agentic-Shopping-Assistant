"""LangGraph pipeline: supervisor routing, parallel agent stages, checkpointed memory.

Flow per turn:
    START → supervisor (LLM plan, dynamic scheduling)
          ├─ general        → assistant (tool-calling agent, streamed)
          └─ product_search → [profile?, recall] → [rerank ‖ inventory]
                              → aggregate → [marketing?] → respond (streamed)
"""

from __future__ import annotations

import time
from typing import Annotated, Any, TypedDict

import structlog
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agents import (
    ChatAgent,
    InventoryAgent,
    MarketingCopyAgent,
    ProductRecAgent,
    SupervisorAgent,
    UserProfileAgent,
)
from agents.models import build_llm
from agents.language import language_directive
from agents.product_rec_agent import recall_products
from agents.structured import text_of
from config import get_settings
from models.schemas import CopyItem, InventoryItem, Product, SearchParams, UserProfile
from services.ab_test import ABTestEngine

logger = structlog.get_logger()
settings = get_settings()

# Module-level singletons; tests replace their LLM attributes with stubs.
supervisor_agent = SupervisorAgent()
profile_agent = UserProfileAgent()
rec_agent = ProductRecAgent()
inventory_agent = InventoryAgent()
copy_agent = MarketingCopyAgent()
chat_agent = ChatAgent()
ab_engine = ABTestEngine()

reply_llm = build_llm(temperature=0.7, max_tokens=512)

REPLY_SYSTEM = """You are a helpful shopping assistant closing a recommendation turn.
Summarise the picks in 2-3 warm, concrete sentences: name the top product, why it fits
(price, rating, tags), and one alternative. Mention stock only if something is low."""


def merge_dicts(left: dict | None, right: dict | None) -> dict:
    return {**(left or {}), **(right or {})}


class GraphState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    user_id: str
    query: str
    language: str
    plan: dict[str, Any]
    profile: UserProfile | None
    candidates: list[Product]
    ranked: list[Product]
    inventory: list[InventoryItem]
    available_ids: list[str]
    copies: list[CopyItem]
    final_products: list[Product]
    reply: str
    timings: Annotated[dict[str, float], merge_dicts]


# ── streaming helpers ─────────────────────────────────────────────────

def emit(event: dict[str, Any]) -> None:
    """Send a custom event to the SSE stream when one is attached."""
    try:
        writer = get_stream_writer()
        if writer:
            writer(event)
    except Exception:  # no active stream (e.g. ainvoke)
        pass


def agent_event(agent: str, status: str, message: str = "") -> None:
    emit({"type": "agent", "agent": agent, "status": status, "message": message})


def elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


# ── nodes ─────────────────────────────────────────────────────────────

async def supervisor_node(state: GraphState) -> dict:
    """Ask the LLM to plan this turn, then reset transient state for a clean run."""
    start = time.perf_counter()
    agent_event("supervisor", "running")

    result = await supervisor_agent.run(
        messages=state.get("messages", []),
        language=state.get("language", "en"),
    )
    plan = result.plan
    variant = ab_engine.assign(state["user_id"])

    agent_event("supervisor", "done", plan.reply)
    emit({
        "type": "plan",
        "intent": plan.intent,
        "reply": plan.reply,
        "search": plan.search.model_dump(),
        "agents": plan.agents,
        "variant": variant,
    })
    emit({"type": "experiment", **ab_engine.info(state["user_id"]).model_dump()})

    return {
        "plan": plan.model_dump(),
        "profile": None,
        "candidates": [],
        "ranked": [],
        "inventory": [],
        "available_ids": [],
        "copies": [],
        "final_products": [],
        "reply": "",
        "timings": {"supervisor": elapsed_ms(start)},
    }


async def assistant_node(state: GraphState) -> dict:
    """Answer a general question with the tool-calling chat agent, streaming tokens."""
    start = time.perf_counter()
    agent_event("assistant", "running")

    text = ""
    async for token in chat_agent.astream(
        state.get("messages", []), state["user_id"], state.get("language", "en")
    ):
        text += token
        emit({"type": "token", "content": token})

    agent_event("assistant", "done")
    return {
        "reply": text,
        "messages": [AIMessage(content=text)],
        "timings": {"assistant": elapsed_ms(start)},
    }


async def profile_node(state: GraphState) -> dict:
    """Build the shopper profile (RFM scores + segment)."""
    start = time.perf_counter()
    agent_event("profile", "running")

    result = await profile_agent.run(user_id=state["user_id"])
    profile = result.profile
    if profile:
        emit({"type": "profile", "profile": profile.model_dump()})
    agent_event("profile", "done", profile.segment if profile else "unavailable")

    return {"profile": profile, "timings": {"profile": elapsed_ms(start)}}


async def recall_node(state: GraphState) -> dict:
    """Filter the catalog with the supervisor's search parameters."""
    start = time.perf_counter()
    agent_event("recommendation", "running", "Searching the catalog…")

    search = SearchParams(**(state.get("plan") or {}).get("search", {}))
    candidates = recall_products(search, limit=settings.max_candidates)

    return {"candidates": candidates, "timings": {"recall": elapsed_ms(start)}}


async def rerank_node(state: GraphState) -> dict:
    """Re-order the candidates with the LLM for this shopper and query."""
    start = time.perf_counter()
    result = await rec_agent.run(
        profile=state.get("profile"),
        products=state.get("candidates", []),
        query=state.get("query", ""),
        k=settings.max_products,
    )
    agent_event("recommendation", "done", f"Top {len(result.products)} picks ready")

    return {"ranked": result.products, "timings": {"rerank": elapsed_ms(start)}}


async def inventory_node(state: GraphState) -> dict:
    """Check stock for every candidate and derive purchase limits."""
    start = time.perf_counter()
    agent_event("inventory", "running")

    result = await inventory_agent.run(products=state.get("candidates", []))
    return {
        "inventory": result.items,
        "available_ids": result.available_ids,
        "timings": {"inventory": elapsed_ms(start)},
    }


async def aggregate_node(state: GraphState) -> dict:
    """Filter the ranking by availability and publish the final product list."""
    start = time.perf_counter()
    ranked = state.get("ranked") or state.get("candidates", [])
    available = set(state.get("available_ids") or [])

    final = [p for p in ranked if not available or p.product_id in available]
    if not final:
        final = ranked
    final = final[: settings.max_products]

    emit({"type": "products", "products": [p.model_dump() for p in final]})
    return {"final_products": final, "timings": {"aggregate": elapsed_ms(start)}}


async def marketing_node(state: GraphState) -> dict:
    """Generate segment-specific copy for the final products."""
    start = time.perf_counter()
    agent_event("copywriting", "running")

    result = await copy_agent.run(
        profile=state.get("profile"),
        products=state.get("final_products", []),
        language=state.get("language", "en"),
    )
    profile = state.get("profile")
    emit({
        "type": "marketing",
        "items": [item.model_dump() for item in result.items],
        "segment": profile.segment if profile else "New",
    })
    agent_event("copywriting", "done", f"{len(result.items)} lines")

    return {"copies": result.items, "timings": {"marketing": elapsed_ms(start)}}


async def respond_node(state: GraphState) -> dict:
    """Publish stock status, then stream the closing reply for shopping turns."""
    start = time.perf_counter()
    agent_event("supervisor", "running", "Writing the summary…")

    final_ids = {p.product_id for p in state.get("final_products", [])}
    stock = [item for item in state.get("inventory", []) if item.product_id in final_ids]
    out_of_stock = [item.name for item in stock if item.status == "out_of_stock"]
    low_stock = [item.name for item in stock if item.status == "low_stock"]
    if out_of_stock:
        summary = f"Not available right now: {', '.join(out_of_stock)}."
    elif low_stock:
        summary = f"Low stock on {', '.join(low_stock)} — they may sell out soon."
    else:
        summary = "All items are in stock and ready to ship."
    emit({
        "type": "inventory",
        "items": [item.model_dump() for item in stock],
        "summary": summary,
    })

    profile = state.get("profile")
    products = "\n".join(
        f"- {p.name} (${p.price:.2f}, {p.rating}★, {', '.join(p.tags)})"
        for p in state.get("final_products", [])
    )
    copies = "\n".join(f"- {c.text}" for c in state.get("copies", []))
    prompt = (
        f"Shopper: {profile.name if profile else 'guest'} "
        f"({profile.segment if profile else 'unknown segment'}).\n"
        f"Request: {state.get('query', '')}\n"
        f"Recommended products:\n{products or 'none'}\n"
        f"Marketing lines already shown to the user:\n{copies or 'none'}"
    )

    text = ""
    async for chunk in reply_llm.astream([
        ("system", f"{REPLY_SYSTEM}\n{language_directive(state.get('language', 'en'))}"),
        ("user", prompt),
    ]):
        token = text_of(chunk.content)
        if token:
            text += token
            emit({"type": "token", "content": token})

    agent_event("supervisor", "done")
    return {
        "reply": text,
        "messages": [AIMessage(content=text)],
        "timings": {"respond": elapsed_ms(start)},
    }


# ── routing ───────────────────────────────────────────────────────────

def route_supervisor(state: GraphState) -> str | list[str]:
    """Dynamic scheduling: general chat, or the agent fan-out from the plan."""
    plan = state.get("plan") or {}
    if plan.get("intent") != "product_search":
        return "assistant"
    targets = ["recall"]
    if "profile" in plan.get("agents", []):
        targets.append("profile")
    return targets


def route_aggregate(state: GraphState) -> str:
    """Run the copy agent only when the plan asked for it."""
    plan = state.get("plan") or {}
    return "marketing" if "copy" in plan.get("agents", []) else "respond"


def build_graph(checkpointer: Any = None):
    """Compile the pipeline; pass a checkpointer to persist conversation state."""
    graph = StateGraph(GraphState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("assistant", assistant_node)
    graph.add_node("profile", profile_node)
    graph.add_node("recall", recall_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("inventory", inventory_node)
    graph.add_node("aggregate", aggregate_node)
    graph.add_node("marketing", marketing_node)
    graph.add_node("respond", respond_node)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", route_supervisor, ["assistant", "profile", "recall"])
    graph.add_edge("assistant", END)

    for stage in ("profile", "recall"):
        graph.add_edge(stage, "rerank")
        graph.add_edge(stage, "inventory")
    graph.add_edge("rerank", "aggregate")
    graph.add_edge("inventory", "aggregate")

    graph.add_conditional_edges("aggregate", route_aggregate, ["marketing", "respond"])
    graph.add_edge("marketing", "respond")
    graph.add_edge("respond", END)

    return graph.compile(checkpointer=checkpointer or InMemorySaver())
