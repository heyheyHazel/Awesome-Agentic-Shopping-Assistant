"""FastAPI entry point: SSE chat streaming plus profile and experiment APIs."""

from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage

from agents import UserProfileAgent
from agents.user_profile_agent import SEGMENTS
from config import get_settings
from data.users import USERS
from models.schemas import (
    ChatRequest,
    ExperimentInfo,
    ProfileResponse,
    RecommendRequest,
    RecommendationResponse,
    UserSummary,
)
from orchestrator.graph import ab_engine, build_graph

logger = structlog.get_logger()
settings = get_settings()

graph: Any = None
profile_agent = UserProfileAgent()


@asynccontextmanager
async def lifespan(_: FastAPI):
    global graph
    graph = build_graph()
    logger.info("app.startup", model=settings.llm_model)
    yield


app = FastAPI(
    title="Multi-Agent Shopping Assistant",
    description="Supervisor routing plus profile, recall, rerank, inventory and copy agents.",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy", "model": settings.llm_model}


@app.get("/api/v1/users", response_model=list[UserSummary])
async def list_users() -> list[UserSummary]:
    """Demo shoppers available in the UI."""
    return [
        UserSummary(user_id=u.user_id, name=u.name, email=u.email, tier=u.tier)
        for u in USERS.values()
    ]


@app.get("/api/v1/users/{user_id}/profile", response_model=ProfileResponse)
async def get_profile(user_id: str) -> ProfileResponse:
    """RFM profile rendered in the dashboard side panel."""
    result = await profile_agent.run(user_id=user_id)
    if not result.profile:
        raise HTTPException(status_code=404, detail="Unknown user")
    return ProfileResponse(profile=result.profile, segments=SEGMENTS)


@app.get("/api/v1/experiments", response_model=ExperimentInfo)
async def get_experiment(user_id: str | None = None) -> ExperimentInfo:
    """Current A/B state, including the variant of a given user."""
    return ab_engine.info(user_id)


@app.post("/api/v1/experiments/outcome")
async def record_outcome(variant: str, success: bool) -> dict:
    """Feed a conversion event into the Thompson Sampling posterior."""
    ab_engine.record_outcome(variant, success)
    return {"status": "recorded"}


@app.post("/api/v1/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    """Chat endpoint streaming SSE events.

    Event order: session → agent/plan/profile/products/marketing/inventory → token → done.
    """
    thread_id = request.thread_id or str(uuid.uuid4())

    async def stream():
        yield _sse("session", {"thread_id": thread_id})
        start = time.perf_counter()
        timings: dict[str, float] = {}
        try:
            config = {"configurable": {"thread_id": thread_id}}
            inputs = {
                "messages": [HumanMessage(content=request.message)],
                "query": request.message,
                "user_id": request.user_id,
            }
            async for mode, chunk in graph.astream(
                inputs, config=config, stream_mode=["custom", "updates"]
            ):
                if mode == "custom":
                    yield _sse(chunk.get("type", "message"), chunk)
                elif isinstance(chunk, dict):
                    for delta in chunk.values():
                        if isinstance(delta, dict) and isinstance(delta.get("timings"), dict):
                            timings.update(delta["timings"])
            yield _sse(
                "done",
                {"latency_ms": round((time.perf_counter() - start) * 1000, 1), "timings": timings},
            )
        except Exception as exc:
            logger.error("chat.failed", error=str(exc))
            yield _sse("error", {"message": "Something went wrong. Please try again."})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/v1/recommend", response_model=RecommendationResponse)
async def recommend(request: RecommendRequest) -> RecommendationResponse:
    """Non-streaming variant of /chat, handy for API clients and Swagger."""
    start = time.perf_counter()
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content=request.query)],
            "query": request.query,
            "user_id": request.user_id,
        },
        config={"configurable": {"thread_id": f"recommend-{uuid.uuid4()}"}},
    )
    final_ids = {p.product_id for p in result.get("final_products", [])}
    return RecommendationResponse(
        user_id=request.user_id,
        reply=result.get("reply", ""),
        products=result.get("final_products", []),
        copies=result.get("copies", []),
        inventory=[i for i in result.get("inventory", []) if i.product_id in final_ids],
        experiment=ab_engine.info(request.user_id),
        latency_ms=round((time.perf_counter() - start) * 1000, 1),
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
