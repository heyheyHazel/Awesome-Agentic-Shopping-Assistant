"""FastAPI entry point: SSE chat streaming plus profile and experiment APIs.

One tool-calling agent handles every turn. The endpoint owns the transport concerns
(session id, A/B bucketing, event framing, timings); the agent owns the reasoning.
"""

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
from fastapi.staticfiles import StaticFiles

from shopping_assistant.agent.shopping_agent import ShoppingAgent
from shopping_assistant.catalog import SOURCE, USERS
from shopping_assistant.domain.models import (
    ChatRequest,
    ExperimentInfo,
    ProfileResponse,
    RecommendationResponse,
    RecommendRequest,
    UserSummary,
)
from shopping_assistant.domain.rfm import SEGMENTS, build_profile
from shopping_assistant.services.ab_test import ABTestEngine
from shopping_assistant.settings import get_settings

logger = structlog.get_logger()
settings = get_settings()

ab_engine = ABTestEngine()

# Built on first use: constructing it eagerly would make importing this module
# require LLM credentials, which breaks tooling and any request-independent test.
_agent: ShoppingAgent | None = None


def get_agent() -> ShoppingAgent:
    global _agent
    if _agent is None:
        _agent = ShoppingAgent()
    return _agent


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("app.startup", model=settings.llm_model, data_source=SOURCE, products=len(USERS))
    yield
    logger.info("app.shutdown")


app = FastAPI(
    title="Awesome Agentic Shopping Assistant",
    description="An Agentic shopping agent over a 23k-product ShopSimulator catalogue.",
    version="3.0.0",
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


def _track_timings(event: dict[str, Any], pending: dict[str, float], timings: dict[str, float]) -> None:
    """Turn the tools' running/done events into a per-tool duration breakdown."""
    if event.get("type") != "tool":
        return
    name, status = event.get("tool"), event.get("status")
    if not name:
        return
    if status == "running":
        pending[name] = time.perf_counter()
    elif name in pending:
        timings[name] = round((time.perf_counter() - pending.pop(name)) * 1000, 1)


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy", "model": settings.llm_model}


@app.get("/api/v1/meta")
async def meta() -> dict:
    """Active catalog source and currency, used by the UI for price formatting."""
    from shopping_assistant.domain.currency import currency_code, currency_symbol

    return {
        "data_source": SOURCE,
        "currency": currency_code(),
        "currency_symbol": currency_symbol(),
    }


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
    profile = build_profile(user_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Unknown user")
    return ProfileResponse(profile=profile, segments=SEGMENTS)


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

    Event order: session → experiment → tool/products/profile/inventory → token → done.
    """
    thread_id = request.thread_id or str(uuid.uuid4())

    async def stream():
        yield _sse("session", {"thread_id": thread_id})
        yield _sse("experiment", ab_engine.info(request.user_id).model_dump())

        start = time.perf_counter()
        pending: dict[str, float] = {}
        timings: dict[str, float] = {}
        try:
            async for event in get_agent().astream(
                request.message,
                user_id=request.user_id,
                language=request.language,
                thread_id=thread_id,
            ):
                _track_timings(event, pending, timings)
                yield _sse(event.get("type", "message"), event)
            yield _sse(
                "done",
                {"latency_ms": round((time.perf_counter() - start) * 1000, 1), "timings": timings},
            )
        except Exception as exc:  # noqa: BLE001
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
    products: list = []
    inventory: list = []
    reply = ""

    async for event in get_agent().astream(
        request.query,
        user_id=request.user_id,
        language=request.language,
        thread_id=f"recommend-{uuid.uuid4()}",
    ):
        kind = event.get("type")
        if kind == "products":
            products = event["products"]
        elif kind == "inventory":
            inventory = event["items"]
        elif kind == "token":
            reply += event["content"]

    return RecommendationResponse(
        user_id=request.user_id,
        reply=reply,
        products=products,
        inventory=inventory,
        experiment=ab_engine.info(request.user_id),
        latency_ms=round((time.perf_counter() - start) * 1000, 1),
    )


def mount_frontend(app: FastAPI) -> None:
    """Serve the built frontend, so one process runs the whole product.

    The frontend calls the API on relative paths, so serving both from this origin
    needs no configuration and no CORS. A missing build is not an error: during
    development Vite serves the UI and proxies /api here.

    Mounted last on purpose — a mount at "/" swallows every path registered after it.
    """
    dist = settings.frontend_dist
    if not (dist / "index.html").exists():
        logger.info(
            "frontend.not_built",
            path=str(dist),
            hint="run `npm run build` in frontend/ to serve the UI from here",
        )
        return

    app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
    logger.info("frontend.served", path=str(dist))


mount_frontend(app)


def run() -> None:
    """Entry point for the `shopping-assistant` console script."""
    uvicorn.run("shopping_assistant.api.app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    run()
