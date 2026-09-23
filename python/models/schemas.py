"""Shared data models for the multi-agent pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ── Domain models ─────────────────────────────────────────────────────


class Product(BaseModel):
    product_id: str
    name: str
    category: str
    price: float
    brand: str = ""
    stock: int = 0
    rating: float = 0.0
    rating_count: int = 0
    tags: list[str] = Field(default_factory=list)


class RFM(BaseModel):
    """Raw behavior metrics plus 0-1 scores computed by the profile agent."""

    recency_days: int
    orders: int
    lifetime_value: float
    recency: float
    frequency: float
    monetary: float
    overall: float


class UserProfile(BaseModel):
    user_id: str
    name: str
    email: str
    tier: str
    segment: str
    rfm: RFM
    preferred_categories: list[str] = Field(default_factory=list)
    price_range: tuple[float, float] = (0.0, 500.0)


class InventoryItem(BaseModel):
    product_id: str
    name: str
    stock: int
    status: Literal["in_stock", "low_stock", "out_of_stock"]
    purchase_limit: int | None = None


# ── LLM structured outputs ────────────────────────────────────────────


class SearchParams(BaseModel):
    """Catalog filters extracted by the supervisor."""

    keywords: list[str] = Field(
        default_factory=list,
        description="Short lowercase search terms, e.g. ['running shoes', 'training']",
    )
    category: str = Field("", description="Catalog category when obvious, else empty string")
    brand: str = ""
    min_price: float | None = None
    max_price: float | None = None


class SupervisorPlan(BaseModel):
    """Routing decision for one conversation turn."""

    intent: Literal["product_search", "general"] = Field(
        description="product_search when the shopper wants product recommendations, otherwise general"
    )
    reply: str = Field(
        description="One-sentence acknowledgement shown immediately, e.g. 'Got it! Here are the best running shoes for you.'"
    )
    search: SearchParams = Field(default_factory=SearchParams)
    agents: list[Literal["profile", "recall", "rerank", "inventory", "copy"]] = Field(
        default_factory=list,
        description=(
            "Pipeline agents to run. For product_search always include recall, rerank, inventory; "
            "add profile when personalization matters; add copy for marketing text. Empty for general."
        ),
    )


class ProductRanking(BaseModel):
    product_ids: list[str] = Field(description="Product IDs ordered from best to worst match")


class CopyItem(BaseModel):
    product_id: str
    text: str


class CopySet(BaseModel):
    items: list[CopyItem]


# ── Agent results ─────────────────────────────────────────────────────


class AgentResult(BaseModel):
    agent_name: str
    success: bool = True
    latency_ms: float = 0.0
    error: str | None = None


class UserProfileResult(AgentResult):
    agent_name: str = "profile"
    profile: UserProfile | None = None


class SupervisorResult(AgentResult):
    agent_name: str = "supervisor"
    plan: "SupervisorPlan | None" = None


class ProductRecResult(AgentResult):
    agent_name: str = "rerank"
    products: list[Product] = Field(default_factory=list)


class CopyResult(AgentResult):
    agent_name: str = "copy"
    items: list[CopyItem] = Field(default_factory=list)


class InventoryResult(AgentResult):
    agent_name: str = "inventory"
    items: list[InventoryItem] = Field(default_factory=list)
    available_ids: list[str] = Field(default_factory=list)


# ── API models ────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    user_id: str = "U001"
    message: str
    thread_id: str | None = None


class RecommendRequest(BaseModel):
    user_id: str = "U001"
    query: str


class UserSummary(BaseModel):
    user_id: str
    name: str
    email: str
    tier: str


class ProfileResponse(BaseModel):
    profile: UserProfile
    segments: list[str]


class VariantStats(BaseModel):
    name: str
    label: str
    trials: int
    conversion_rate: float


class ExperimentInfo(BaseModel):
    experiment_id: str
    name: str
    variant: str | None = None
    variants: list[VariantStats] = Field(default_factory=list)
    winner: str = ""


class RecommendationResponse(BaseModel):
    user_id: str
    reply: str
    products: list[Product] = Field(default_factory=list)
    copies: list[CopyItem] = Field(default_factory=list)
    inventory: list[InventoryItem] = Field(default_factory=list)
    experiment: ExperimentInfo | None = None
    latency_ms: float = 0.0
