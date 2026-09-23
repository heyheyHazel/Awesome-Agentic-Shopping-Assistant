"""LangGraph pipeline: supervisor routing, deterministic filtering, one ranking call.

Flow per turn:
    START → supervisor (LLM plan: intent + retrieval parameters)
          ├─ general        → assistant (tool-calling agent, streamed)
          └─ product_search → profile → recall → inventory → filter
                              → recommend (LLM: ranking + pitch lines in one call)
                              → respond (streamed answer, no LLM)

Two LLM round trips per shopping turn. Query understanding and ranking both need
the model; everything else (recall, stock, filtering, composing the reply) is
deterministic and costs no latency, so it is not worth a separate call.
"""

from __future__ import annotations

import time
from typing import Annotated, Any, TypedDict

import structlog
from langchain_core.messages import AIMessage, AnyMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agents import (
    ChatAgent,
    InventoryAgent,
    ProductRecAgent,
    SupervisorAgent,
    UserProfileAgent,
)
from agents.product_rec_agent import recall_products
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
chat_agent = ChatAgent()
ab_engine = ABTestEngine()

# The reply is assembled from the pitch lines the ranking call already wrote, so
# it costs no extra round trip. Chunked so the UI keeps its streaming feel.
TOKEN_CHUNK = 6


class GraphState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    user_id: str
    query: str
    language: str
    plan: dict[str, Any]
    profile: UserProfile | None
    candidates: list[Product]
    inventory: list[InventoryItem]
    available_ids: list[str]
    final_products: list[Product]
    pitches: list[CopyItem]
    reply: str
    timings: Annotated[dict[str, float], merge_dicts]


def merge_dicts(left: dict | None, right: dict | None) -> dict:
    return {**(left or {}), **(right or {})}


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


def _chunks(text: str) -> list[str]:
    return [text[index:index + TOKEN_CHUNK] for index in range(0, len(text), TOKEN_CHUNK)]


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
        "variant": variant,
    })
    emit({"type": "experiment", **ab_engine.info(state["user_id"]).model_dump()})

    return {
        "plan": plan.model_dump(),
        "profile": None,
        "candidates": [],
        "inventory": [],
        "available_ids": [],
        "final_products": [],
        "pitches": [],
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
    """Build the shopper profile (percentile RFM scores + segment)."""
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
    candidates = recall_products(
        search, query=state.get("query", ""), limit=settings.max_candidates
    )

    return {"candidates": candidates, "timings": {"recall": elapsed_ms(start)}}


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


async def filter_node(state: GraphState) -> dict:
    """Drop anything out of stock before the model spends a call ranking it."""
    start = time.perf_counter()
    available = set(state.get("available_ids") or [])
    candidates = state.get("candidates", [])
    shortlist = [p for p in candidates if p.product_id in available] if available else candidates
    return {"candidates": shortlist, "timings": {"filter": elapsed_ms(start)}}


async def recommend_node(state: GraphState) -> dict:
    """One LLM call: rank the shortlist and write a pitch line for each pick."""
    start = time.perf_counter()
    agent_event("recommendation", "running", "Ranking the best matches…")

    result = await rec_agent.run(
        profile=state.get("profile"),
        products=state.get("candidates", []),
        query=state.get("query", ""),
        language=state.get("language", "en"),
        k=settings.max_products,
    )
    agent_event("recommendation", "done", f"Top {len(result.products)} picks ready")

    # A deliberate empty ranking means "none of these fit the request"; only a
    # failed call falls back to the shortlist order, so degradation still works.
    final = (
        result.products
        if result.success
        else state.get("candidates", [])[: settings.max_products]
    )
    return {
        "final_products": final,
        "pitches": result.pitches,
        "timings": {"recommend": elapsed_ms(start)},
    }


async def respond_node(state: GraphState) -> dict:
    """Publish products and stock status, then stream the pitch lines as the answer."""
    start = time.perf_counter()
    final = state.get("final_products", [])
    emit({"type": "products", "products": [p.model_dump() for p in final]})

    final_ids = {p.product_id for p in final}
    stock = [item for item in state.get("inventory", []) if item.product_id in final_ids]
    if stock:
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

    text = "\n".join(pitch.text.strip() for pitch in state.get("pitches", []) if pitch.text.strip())
    if text:
        agent_event("copywriting", "running")
        for chunk in _chunks(text):
            emit({"type": "token", "content": chunk})
        agent_event("copywriting", "done")

    return {
        "reply": text,
        "messages": [AIMessage(content=text)] if text else [],
        "timings": {"respond": elapsed_ms(start)},
    }


# ── routing ───────────────────────────────────────────────────────────

def route_supervisor(state: GraphState) -> str:
    """General questions go to the tool-calling agent; shopping turns run the pipeline.

    The deterministic stages (profile, recall, inventory, filter) are chained as a
    straight line rather than fanned out. Each costs well under a millisecond while
    an LLM call costs seconds, so parallelising them buys nothing — and a node with
    two incoming edges from different supersteps runs once per superstep, which
    would execute the ranking call twice.
    """
    plan = state.get("plan") or {}
    return "assistant" if plan.get("intent") != "product_search" else "profile"


def route_filter(state: GraphState) -> str:
    """Skip the ranking call entirely when nothing survived the constraints."""
    return "recommend" if state.get("candidates") else "respond"


def build_graph(checkpointer: Any = None):
    """Compile the pipeline; pass a checkpointer to persist conversation state."""
    graph = StateGraph(GraphState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("assistant", assistant_node)
    graph.add_node("profile", profile_node)
    graph.add_node("recall", recall_node)
    graph.add_node("inventory", inventory_node)
    graph.add_node("filter", filter_node)
    graph.add_node("recommend", recommend_node)
    graph.add_node("respond", respond_node)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", route_supervisor, ["assistant", "profile"])
    graph.add_edge("assistant", END)

    graph.add_edge("profile", "recall")
    graph.add_edge("recall", "inventory")
    graph.add_edge("inventory", "filter")
    graph.add_conditional_edges("filter", route_filter, ["recommend", "respond"])
    graph.add_edge("recommend", "respond")
    graph.add_edge("respond", END)

    return graph.compile(checkpointer=checkpointer or InMemorySaver())
