"""Configuration loaded from environment variables (prefix ECOM_)."""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Multi-Agent Shopping Assistant"

    # LLM — any OpenAI-compatible endpoint
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.4
    llm_request_timeout: float = 60.0
    # DeepSeek-style thinking models: disable hidden reasoning for fast structured calls
    # (only enable this for providers that accept the `thinking` parameter)
    llm_disable_thinking: bool = False

    # Pipeline
    max_products: int = 3       # products shown after aggregation
    max_candidates: int = 12    # candidates kept after recall

    # Agent timeouts (seconds)
    agent_timeout_default: float = 8.0
    agent_timeout_llm: float = 25.0
    agent_timeout_chat: float = 45.0

    model_config = {"env_file": ".env", "env_prefix": "ECOM_", "extra": "ignore"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()
