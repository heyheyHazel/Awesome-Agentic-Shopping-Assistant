"""Configuration loaded from environment variables (prefix ECOM_)."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings

# src/shopping_assistant/settings.py -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]

# Runtime artifacts live outside the source tree. Downloaded source data and the
# generated catalogue go under data/; model weights are kept separate because they
# are a different kind of thing (assets to fetch, not data to derive).
# Override with ECOM_DATA_DIR / ECOM_MODELS_DIR.
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_MODELS_DIR = REPO_ROOT / "models"


class Settings(BaseSettings):
    app_name: str = "Agentic Shopping Assistant"

    # Downloaded source data and the generated catalogue / vector index.
    data_dir: Path = DEFAULT_DATA_DIR

    # Model weights (the local ONNX embedding model).
    models_dir: Path = DEFAULT_MODELS_DIR

    # LLM — any OpenAI-compatible endpoint
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.4
    llm_max_tokens: int = 1200
    llm_request_timeout: float = 60.0
    # DeepSeek-style thinking models: disable hidden reasoning for fast structured calls
    # (only enable this for providers that accept the `thinking` parameter)
    llm_disable_thinking: bool = False

    # Data source: auto = generated dataset when present, else the mock catalog
    data_source: str = "auto"

    # Pipeline
    max_products: int = 3       # products shown after ranking
    max_candidates: int = 12    # candidates kept after recall

    # Agent loop budgets: a shopper turn needs a handful of tool calls, not a
    # shopping episode's worth, so this is deliberately tighter than the RL pool.
    max_turns: int = 8
    max_tool_calls: int = 12

    # Currency of the catalog (drives price symbols in the UI and in prompts)
    currency: str = "CNY"

    # A frontend build served by the API, when one exists (see api/app.py).
    frontend_dist: Path = REPO_ROOT / "frontend" / "dist"

    model_config = {
        # Absolute, so the API and the scripts resolve the same .env from any cwd.
        "env_file": str(REPO_ROOT / ".env"),
        "env_prefix": "ECOM_",
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    return Settings()
