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


# ── Search parameters ────────────────────────────────────────────────


class SearchParams(BaseModel):
    """Hard catalogue filters, filled from the model's tool arguments."""

    keywords: list[str] = Field(
        default_factory=list,
        description="Short lowercase search terms, e.g. ['running shoes', 'training']",
    )
    category: str = Field("", description="Catalog category when obvious, else empty string")
    brand: str = ""
    min_price: float | None = None
    max_price: float | None = None


# ── Agent results ─────────────────────────────────────────────────────


class AgentResult(BaseModel):
    agent_name: str
    success: bool = True
    latency_ms: float = 0.0
    error: str | None = None


class UserProfileResult(AgentResult):
    agent_name: str = "profile"
    profile: UserProfile | None = None


class InventoryResult(AgentResult):
    agent_name: str = "inventory"
    items: list[InventoryItem] = Field(default_factory=list)
    available_ids: list[str] = Field(default_factory=list)


# ── API models ────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    user_id: str = "U001"
    message: str
    thread_id: str | None = None
    language: Literal["en", "zh"] = "en"


class RecommendRequest(BaseModel):
    user_id: str = "U001"
    query: str
    language: Literal["en", "zh"] = "en"


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
    inventory: list[InventoryItem] = Field(default_factory=list)
    experiment: ExperimentInfo | None = None
    latency_ms: float = 0.0
