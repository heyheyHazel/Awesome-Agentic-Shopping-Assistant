"""商品推荐 Agent：关键词过滤召回 + LLM 精排。"""

from __future__ import annotations

import json
import random
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import get_settings
from models.schemas import Product, ProductRecResult, UserProfile

from .base_agent import BaseAgent

RERANK_PROMPT = """你是电商推荐排序专家。根据用户需求和候选商品,重新排序并选出最优的{num_items}个商品。

用户画像:
{user_profile}

{search_context}

候选商品:
{candidates}

排序原则:
1. 与用户搜索关键词/类目匹配的商品优先
2. 价格在用户可接受范围内
3. 保证类目多样性(相邻商品尽量不同类目)
4. 新品适当加权

请输出商品ID列表(JSON数组),按推荐优先级排序:
["product_id_1", "product_id_2", ...]

只输出JSON数组,不要其他内容。"""

MOCK_PRODUCTS = [
    Product(product_id="P001", name="iPhone 16 Pro", category="手机", price=7999, brand="Apple", seller_id="S01", stock=500, tags=["旗舰", "新品"]),
    Product(product_id="P002", name="华为 Mate 70", category="手机", price=5999, brand="华为", seller_id="S02", stock=300, tags=["旗舰", "国产"]),
    Product(product_id="P003", name="AirPods Pro 3", category="耳机", price=1899, brand="Apple", seller_id="S01", stock=1000, tags=["降噪", "无线"]),
    Product(product_id="P004", name="Sony WH-1000XM6", category="耳机", price=2499, brand="Sony", seller_id="S03", stock=200, tags=["头戴", "降噪"]),
    Product(product_id="P005", name="iPad Air M3", category="平板", price=4799, brand="Apple", seller_id="S01", stock=400, tags=["学习", "办公"]),
    Product(product_id="P006", name="小米平板7 Pro", category="平板", price=2499, brand="小米", seller_id="S04", stock=600, tags=["性价比", "娱乐"]),
    Product(product_id="P007", name="Anker 140W充电器", category="配件", price=399, brand="Anker", seller_id="S05", stock=2000, tags=["快充", "便携"]),
    Product(product_id="P008", name="机械革命极光X", category="笔记本", price=6999, brand="机械革命", seller_id="S06", stock=150, tags=["游戏", "高性能"]),
    Product(product_id="P009", name="戴尔U2724D显示器", category="显示器", price=3299, brand="Dell", seller_id="S07", stock=80, tags=["4K", "办公"]),
    Product(product_id="P010", name="罗技MX Master 3S", category="配件", price=749, brand="罗技", seller_id="S08", stock=500, tags=["无线", "办公"]),
    Product(product_id="P011", name="三星980 Pro 2TB", category="存储", price=1199, brand="三星", seller_id="S09", stock=300, tags=["SSD", "高速"]),
    Product(product_id="P012", name="绿联氮化镓65W", category="配件", price=129, brand="绿联", seller_id="S10", stock=5000, tags=["快充", "性价比"]),
    Product(product_id="P013", name="Apple Watch Ultra 3", category="穿戴", price=5999, brand="Apple", seller_id="S01", stock=200, tags=["运动", "健康"]),
    Product(product_id="P014", name="大疆Mini 4 Pro", category="无人机", price=4788, brand="大疆", seller_id="S11", stock=100, tags=["航拍", "便携"]),
    Product(product_id="P015", name="Switch 2", category="游戏机", price=2499, brand="Nintendo", seller_id="S12", stock=50, tags=["新品", "游戏"]),
    # ── 美妆护肤 ──
    Product(product_id="P016", name="YSL小金条口红 #21", category="美妆", price=320, brand="YSL", seller_id="S13", stock=800, tags=["口红", "热卖", "显白"]),
    Product(product_id="P017", name="兰蔻持妆粉底液", category="美妆", price=480, brand="兰蔻", seller_id="S14", stock=600, tags=["粉底", "持久", "遮瑕"]),
    Product(product_id="P018", name="MAC生姜高光", category="美妆", price=280, brand="MAC", seller_id="S15", stock=400, tags=["高光", "修容", "自然"]),
    Product(product_id="P019", name="SK-II 神仙水 230ml", category="护肤", price=1370, brand="SK-II", seller_id="S16", stock=200, tags=["精华", "保湿", "高端"]),
    Product(product_id="P020", name="完美日记动物眼影盘", category="美妆", price=129, brand="完美日记", seller_id="S17", stock=1500, tags=["眼影", "国货", "性价比"]),
    Product(product_id="P021", name="欧莱雅紫熨斗眼霜", category="护肤", price=259, brand="欧莱雅", seller_id="S18", stock=700, tags=["眼霜", "抗皱", "淡纹"]),
    Product(product_id="P022", name="薇诺娜舒敏保湿霜", category="护肤", price=168, brand="薇诺娜", seller_id="S19", stock=900, tags=["面霜", "敏感肌", "修护"]),
    # ── 服饰鞋包 ──
    Product(product_id="P023", name="ZARA春夏碎花连衣裙", category="女装", price=299, brand="ZARA", seller_id="S20", stock=350, tags=["连衣裙", "碎花", "春夏"]),
    Product(product_id="P024", name="优衣库防晒衣UV Cut", category="女装", price=199, brand="优衣库", seller_id="S21", stock=1200, tags=["防晒", "轻薄", "基础款"]),
    Product(product_id="P025", name="SW一字带高跟鞋", category="鞋靴", price=3200, brand="Stuart Weitzman", seller_id="S22", stock=80, tags=["高跟鞋", "经典", "通勤"]),
    Product(product_id="P026", name="百丽乐福鞋平底", category="鞋靴", price=599, brand="百丽", seller_id="S23", stock=450, tags=["平底", "乐福鞋", "百搭"]),
    Product(product_id="P027", name="Coach Tabby 26 单肩包", category="箱包", price=3500, brand="Coach", seller_id="S24", stock=120, tags=["单肩包", "轻奢", "通勤"]),
    # ── 个护健康 ──
    Product(product_id="P028", name="戴森Airwrap多功能美发器", category="个护", price=3699, brand="戴森", seller_id="S25", stock=180, tags=["卷发棒", "吹风", "高端"]),
    Product(product_id="P029", name="飞利浦电动牙刷HX9352", category="个护", price=699, brand="飞利浦", seller_id="S26", stock=500, tags=["电动牙刷", "声波", "美白"]),
    # ── 食品饮料 ──
    Product(product_id="P030", name="三顿半精品速溶咖啡24颗", category="食品", price=189, brand="三顿半", seller_id="S27", stock=3000, tags=["咖啡", "速溶", "精品"]),
    Product(product_id="P031", name="良品铺子坚果大礼包", category="食品", price=99, brand="良品铺子", seller_id="S28", stock=5000, tags=["坚果", "零食", "礼盒"]),
    Product(product_id="P032", name="农夫山泉NFC橙汁300mlx12", category="饮料", price=79, brand="农夫山泉", seller_id="S29", stock=8000, tags=["果汁", "NFC", "健康"]),
    # ── 家电家居 ──
    Product(product_id="P033", name="戴森V15吸尘器", category="家电", price=4990, brand="戴森", seller_id="S25", stock=100, tags=["吸尘器", "无线", "除螨"]),
    Product(product_id="P034", name="米家扫拖机器人2", category="家电", price=1599, brand="小米", seller_id="S04", stock=400, tags=["扫地机器人", "智能", "扫拖一体"]),
    Product(product_id="P035", name="网易严选乳胶枕", category="家居", price=199, brand="网易严选", seller_id="S30", stock=1000, tags=["枕头", "乳胶", "护颈"]),
    # ── 运动户外 ──
    Product(product_id="P036", name="Nike Air Zoom Pegasus 40", category="运动", price=899, brand="Nike", seller_id="S31", stock=600, tags=["跑鞋", "缓震", "训练"]),
    Product(product_id="P037", name="Lululemon Align 瑜伽裤", category="运动", price=750, brand="Lululemon", seller_id="S32", stock=300, tags=["瑜伽裤", "运动", "舒适"]),
    Product(product_id="P038", name="迪卡侬折叠露营椅", category="户外", price=149, brand="迪卡侬", seller_id="S33", stock=2000, tags=["露营", "折叠", "便携"]),
]


class ProductRecAgent(BaseAgent):
    def __init__(self):
        settings = get_settings()
        super().__init__(
            name="product_rec",
            timeout=settings.agent_timeout_product_rec,
        )
        self.llm = ChatOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            temperature=0.3,
            max_tokens=512,  # type: ignore[reportCallIssue]
        )


    async def _execute(self, **kwargs: Any) -> ProductRecResult:
        user_profile: UserProfile | None = kwargs.get("user_profile")
        num_items: int = kwargs.get("num_items", 10)
        keyword: str = kwargs.get("keyword", "")
        category: str = kwargs.get("category", "")
        brand: str = kwargs.get("brand", "")
        skip_rerank: bool = kwargs.get("skip_rerank", False)

        candidates = await self._recall(user_profile, num_items * 3, keyword=keyword, category=category, brand=brand)
        if skip_rerank:
            return ProductRecResult(
                success=True,
                products=candidates[:num_items],
                recall_strategy="keyword_filter",
                data={"candidate_count": len(candidates), "reranked": 0},
                confidence=0.7,
            )
        ranked_ids = await self._rerank(user_profile, candidates, num_items, keyword=keyword, category=category)

        id_to_product = {p.product_id: p for p in candidates}
        final_products = []
        for pid in ranked_ids:
            if pid in id_to_product:
                final_products.append(id_to_product[pid])
        if len(final_products) < num_items:
            for p in candidates:
                if p.product_id not in ranked_ids:
                    final_products.append(p)
                    if len(final_products) >= num_items:
                        break

        return ProductRecResult(
            success=True,
            products=final_products[:num_items],
            recall_strategy="keyword_filter+llm_rerank",
            data={"candidate_count": len(candidates), "reranked": len(ranked_ids)},
            confidence=0.8,
        )

    async def _recall(
        self, profile: UserProfile | None, limit: int,
        keyword: str = "", category: str = "", brand: str = "",
    ) -> list[Product]:
        """Recall from the catalog with keyword/category/brand filtering."""
        candidates = list(MOCK_PRODUCTS)

        # ── Keyword / category / brand filtering ──
        search_term = (keyword or category or brand).strip().lower()
        if search_term:
            filtered = []
            for p in candidates:
                name_lower = p.name.lower()
                cat_lower = p.category.lower()
                brand_lower = p.brand.lower()
                if (search_term in name_lower
                        or search_term in cat_lower
                        or search_term in brand_lower
                        or any(search_term in t.lower() for t in p.tags)):
                    filtered.append(p)
            if filtered:
                candidates = filtered

        if profile and profile.preferred_categories:
            preferred = set(profile.preferred_categories)
            candidates.sort(
                key=lambda p: (p.category in preferred, p.stock > 0, random.random()),
                reverse=True,
            )

        return candidates[:limit]

    async def _rerank(
        self, profile: UserProfile | None, candidates: list[Product], num_items: int,
        keyword: str = "", category: str = "",
    ) -> list[str]:
        if not profile and not keyword and not category:
            return [p.product_id for p in candidates[:num_items]]

        profile_summary = {
            "segments": [s.value for s in profile.segments] if profile else [],
            "preferred_categories": profile.preferred_categories if profile else [],
            "price_range": list(profile.price_range) if profile else [0, 10000],
        }

        search_context = ""
        search_term = keyword or category
        if search_term:
            search_context = f"用户当前搜索: {search_term}\n请优先推荐与该搜索词匹配的商品。"

        candidate_summary = [
            {"id": p.product_id, "name": p.name, "category": p.category, "price": p.price, "tags": p.tags}
            for p in candidates
        ]
        prompt = RERANK_PROMPT.format(
            num_items=num_items,
            user_profile=json.dumps(profile_summary, ensure_ascii=False),
            search_context=search_context,
            candidates=json.dumps(candidate_summary, ensure_ascii=False),
        )
        messages = [
            SystemMessage(content="你是电商推荐排序专家。"),
            HumanMessage(content=prompt),
        ]
        response = await self.llm.ainvoke(messages)
        try:
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            return json.loads(raw)
        except (json.JSONDecodeError, IndexError):
            return [p.product_id for p in candidates[:num_items]]
