"""Configuration for the shoprl stack (environment prefix ``SHOPRL_``)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings

REPO_ROOT = Path(__file__).resolve().parents[2]


class ShopRLSettings(BaseSettings):
    """Paths, rollout budgets and backend endpoints, all overridable from ``.env``."""

    data_dir: Path = REPO_ROOT / "data"
    models_dir: Path = REPO_ROOT / "models"
    runs_dir: Path = REPO_ROOT / "runs"

    # Point at the official ShopSimulator Flask service to roll out against the
    # reference environment; empty means use the in-process reference environment.
    env_url: str = ""
    env_pool_capacity: int = 8
    # Show the shopper's persona on reset. Only 4,666 of 23,421 tasks carry one.
    persona: bool = False

    max_model_turns: int = 30
    max_tool_calls: int = 60
    max_context_tokens: int = 16384
    max_response_tokens: int = 1024
    context_keep_tool_results: int = 3

    model_config = {
        "env_file": str(REPO_ROOT / ".env"),
        "env_prefix": "SHOPRL_",
        "extra": "ignore",
    }


@lru_cache
def get_settings() -> ShopRLSettings:
    return ShopRLSettings()
