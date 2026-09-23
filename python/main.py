"""
Multi-Agent E-Commerce Recommendation System — FastAPI Entry Point

Endpoints:
  POST /api/v1/recommend          - 获取个性化推荐
  POST /api/v1/recommend/graph    - 通过LangGraph pipeline推荐
  POST /api/v1/chat               - 聊天式推荐 (SSE streaming)
  GET  /api/v1/experiments        - 查看A/B实验状态
  GET  /api/v1/metrics            - 查看系统监控指标
  GET  /health                    - 健康检查
"""

from __future__ import annotations

import json as json_module
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from contextlib import asynccontextmanager

import structlog
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI

from config import get_settings
from models.schemas import ChatRequest, RecommendationRequest, RecommendationResponse
from agents import ChatAgent
from orchestrator.supervisor import SupervisorOrchestrator
from orchestrator.graph import build_recommendation_graph
from services.ab_test import ABTestEngine
from services.metrics import MetricsCollector

logger = structlog.get_logger()
settings = get_settings()


ab_engine = ABTestEngine()
metrics_collector = MetricsCollector()
supervisor = SupervisorOrchestrator(ab_engine=ab_engine)
chat_agent = ChatAgent(supervisor=supervisor)
rec_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rec_graph
    rec_graph = build_recommendation_graph()
    logger.info("app.startup", model=settings.llm_model)
    yield
    logger.info("app.shutdown")


app = FastAPI(
    title="Multi-Agent E-Commerce Recommendation System",
    description="用户画像Agent + 商品推荐Agent + 营销文案Agent + 库存决策Agent，并行+聚合模式",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "healthy", "model": settings.llm_model}


@app.post("/api/v1/recommend", response_model=RecommendationResponse)
async def recommend(request: RecommendationRequest):
    """使用Supervisor编排器进行推荐 (生产推荐用法)"""
    response = await supervisor.recommend(request)
    _collect_metrics(response)
    return response


@app.post("/api/v1/recommend/graph")
async def recommend_via_graph(request: RecommendationRequest):
    """使用LangGraph状态图进行推荐 (展示LangGraph能力)"""
    if not rec_graph:
        return {"error": "Graph not initialized"}
    state = {
        "user_id": request.user_id,
        "scene": request.scene,
        "num_items": request.num_items,
        "context": request.context,
    }
    result = await rec_graph.ainvoke(state)
    return {
        "request_id": result.get("request_id"),
        "user_id": result.get("user_id"),
        "products": [p.model_dump() for p in result.get("final_products", [])],
        "marketing_copies": result.get("marketing_copies", []),
        "experiment_group": result.get("experiment_group", "control"),
        "total_latency_ms": round(result.get("total_latency_ms", 0), 1),
    }


@app.post("/api/v1/chat")
async def chat(request: ChatRequest):
    """聊天式推荐 — runs ChatAgent for intent + products, then streams reply via SSE.

    SSE event flow: products → token (×N) → done
    """
    result = await chat_agent.run(
        user_id=request.user_id,
        messages=[m.model_dump() for m in request.messages],
    )

    intake_products = getattr(result, "products", []) or []
    reply_prompt = getattr(result, "reply_prompt", "") or ""

    async def _event_stream():
        # ── Phase 1: emit products immediately ──
        products_payload = {
            "products": [
                {
                    "product_id": p.product_id,
                    "name": p.name,
                    "price": p.price,
                    "category": p.category,
                    "brand": p.brand,
                    "description": p.description,
                    "image_url": p.image_url,
                }
                for p in intake_products[:6]
            ]
        }
        yield _sse_event("products", products_payload)

        # ── Phase 2: stream reply ──
        if result.success:
            # Agent succeeded — stream LLM tokens for a natural reply
            try:
                stream_llm = ChatOpenAI(
                    api_key=settings.llm_api_key,
                    base_url=settings.llm_base_url,
                    model=settings.llm_model,
                    temperature=0.7,
                    max_tokens=1024,  # type: ignore[reportCallIssue]
                    streaming=True,
                )
                async for chunk in stream_llm.astream(reply_prompt):
                    token = chunk.content if hasattr(chunk, "content") else str(chunk)
                    if token:
                        yield _sse_event("token", {"content": token})
            except Exception as exc:
                logger.error("chat.stream_error", error=str(exc))
                yield _sse_event("token", {"content": "\n\n抱歉，回复生成过程中出现问题，请重试。"})
        else:
            # Agent failed — emit fallback text directly (skip another API call)
            yield _sse_event("token", {"content": reply_prompt})

        # ── Phase 3: done ──
        yield _sse_event("done", {})

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _sse_event(event: str, data: dict) -> str:
    """Format a single SSE event with JSON data payload."""
    return f"event: {event}\ndata: {json_module.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/api/v1/experiments")
async def get_experiments():
    """查看所有A/B实验状态"""
    experiments = {}
    for exp_id, exp in ab_engine.experiments.items():
        experiments[exp_id] = {
            "name": exp.name,
            "enabled": exp.enabled,
            "groups": [
                {
                    "name": g.name,
                    "weight": g.weight,
                    "config": g.config,
                    "successes": g.successes,
                    "failures": g.failures,
                }
                for g in exp.groups
            ],
            "stats": ab_engine.get_stats(exp_id),
        }
    return experiments


@app.get("/api/v1/metrics")
async def get_metrics():
    """查看系统监控指标"""
    return {
        "agents": metrics_collector.get_agent_stats(),
        "business": metrics_collector.get_business_stats(),
    }


@app.post("/api/v1/experiments/{experiment_id}/outcome")
async def record_outcome(experiment_id: str, group: str, success: bool):
    """记录A/B测试结果,更新Thompson Sampling"""
    ab_engine.record_outcome(experiment_id, group, success)
    return {"status": "recorded"}


def _collect_metrics(response: RecommendationResponse):
    for name, result in response.agent_results.items():
        metrics_collector.record_agent_call(
            agent_name=name,
            success=result.success,
            latency_ms=result.latency_ms,
        )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
