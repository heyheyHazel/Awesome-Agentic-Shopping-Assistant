"""ChatAgent：解析多轮对话中的购物意图，并委托 Supervisor 生成推荐。

Agent 本身不负责流式输出，流式回复由 FastAPI 路由在 Agent 返回后完成。
"""

from __future__ import annotations

import json
from typing import Any, TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import get_settings
from models.schemas import (
    AgentResult,
    ChatResult,
    Product,
    RecommendationRequest,
    ShoppingIntent,
)

from .base_agent import BaseAgent

if TYPE_CHECKING:
    from orchestrator.supervisor import SupervisorOrchestrator

INTENT_PARSE_PROMPT = """你是一个电商客服意图分析专家。根据用户的聊天记录，提取购物意图。

输出严格JSON格式（不要markdown包裹）:
{
  "category": "商品类别(如口红、手机、耳机),没有则为空字符串",
  "brand": "偏好品牌,没有则为空字符串",
  "price_min": 最低预算(float,默认0),
  "price_max": 最高预算(float,默认10000),
  "keyword": "核心搜索关键词,没有则为空字符串",
  "intent_type": "product_search 或 general_question",
  "num_items": 期望商品数量(默认6)
}

判断规则:
- 如果用户询问商品、推荐、购买、想买、找一款等 → intent_type="product_search"
- 如果用户只是闲聊、打招呼、问天气等 → intent_type="general_question"
- 提取类别时尽量使用通用电商类目名称"""

REPLY_BUILD_PROMPT = """你是一个友好的电商导购助手。根据以下信息，给用户一个自然、有帮助的回复。

用户意图: {intent_type}
商品类别: {category}
{products_summary}

请用中文回复，要自然亲切，简要介绍推荐的商品，并询问用户是否还有其他需求。"""


class ChatAgent(BaseAgent):
    """Parses shopping intent from chat, delegates to orchestrator for products."""

    def __init__(self, supervisor: SupervisorOrchestrator | None = None):
        settings = get_settings()
        super().__init__(
            name="chat",
            timeout=settings.agent_timeout_chat,
        )
        self._supervisor = supervisor
        self.llm = ChatOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            temperature=0.3,
            max_tokens=1024, # type: ignore
        )

    async def _execute(self, **kwargs: Any) -> ChatResult:
        user_id: str = kwargs["user_id"]
        messages: list[dict] = kwargs.get("messages", [])

        intent = await self._parse_intent(messages)

        if intent.intent_type == "general_question":
            return ChatResult(
                success=True,
                intent=intent,
                reply_prompt=self._build_reply_prompt(intent, []),
                confidence=0.9,
            )

        products, copies = await self._get_recommendations(user_id, intent)
        reply_prompt = self._build_reply_prompt(intent, products)

        return ChatResult(
            success=True,
            intent=intent,
            products=products,
            marketing_copies=list(copies),
            reply_prompt=reply_prompt,
            confidence=0.85,
        )

    async def _parse_intent(self, messages: list[dict]) -> ShoppingIntent:
        """Use LLM to extract structured shopping intent from chat history."""
        chat_text = json.dumps(messages, ensure_ascii=False)
        llm_messages = [
            SystemMessage(content=INTENT_PARSE_PROMPT),
            HumanMessage(content=f"用户聊天记录:\n{chat_text}"),
        ]
        response = await self.llm.ainvoke(llm_messages)
        return self._clean_intent(response.content) # type: ignore

    def _clean_intent(self, raw: str) -> ShoppingIntent:
        """Parse LLM output into ShoppingIntent, with fallback for malformed JSON."""
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
            data = json.loads(cleaned)
        except (json.JSONDecodeError, IndexError):
            return ShoppingIntent()

        return ShoppingIntent(
            category=data.get("category", ""),
            brand=data.get("brand", ""),
            price_min=float(data.get("price_min", 0)),
            price_max=float(data.get("price_max", 10000)),
            keyword=data.get("keyword", ""),
            intent_type=data.get("intent_type", "general_question"),
            num_items=int(data.get("num_items", 6)),
        )

    async def _get_recommendations(
        self, user_id: str, intent: ShoppingIntent
    ) -> tuple[list[Product], list[dict[str, str]]]:
        """Delegate to SupervisorOrchestrator for product recommendations."""
        if self._supervisor is None:
            return [], []

        keyword = intent.keyword or intent.category
        request = RecommendationRequest(
            user_id=user_id,
            scene="chat",
            num_items=intent.num_items,
            context={"keyword": keyword, "category": intent.category, "brand": intent.brand},
        )

        response = await self._supervisor.recommend_chat(request)
        return response.products, response.marketing_copies

    def _build_reply_prompt(
        self, intent: ShoppingIntent, products: list[Product]
    ) -> str:
        """Build a prompt for the streaming reply LLM."""
        if not products:
            if intent.intent_type == "general_question":
                return "用户发来了一条非购物消息，请友好地回复并表示可以帮ta推荐商品。"
            return f"用户想找'{intent.keyword or intent.category}'类商品，但目前没有匹配结果。请礼貌告知并建议用户调整搜索词。"

        product_lines = []
        for i, p in enumerate(products[:6], 1):
            product_lines.append(f"{i}. {p.name} - ¥{p.price:.0f} ({p.brand or '多品牌'})")

        return REPLY_BUILD_PROMPT.format(
            intent_type=intent.intent_type,
            category=intent.category or intent.keyword or "综合",
            products_summary="推荐商品:\n" + "\n".join(product_lines),
        )

    def _fallback(self, latency_ms: float, exc: Exception) -> AgentResult:
        return ChatResult(
            agent_name=self.name,
            success=False,
            latency_ms=latency_ms,
            error=str(exc),
            reply_prompt="抱歉，系统暂时出现问题，请稍后再试。",
            confidence=0.0,
        )
