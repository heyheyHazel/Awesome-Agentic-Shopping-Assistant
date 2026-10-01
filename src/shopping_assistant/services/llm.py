"""The serving-side model client.

One implementation for every OpenAI-compatible provider. Streaming keeps the
typewriter pacing in the UI; the harness treats it as an optional detail and the
training engines ignore it.
"""

from __future__ import annotations

from shopping_assistant.settings import get_settings
from shoprl.harness.backends import OpenAIBackend


def build_backend() -> OpenAIBackend:
    """Create the chat client described by the `ECOM_LLM_*` settings."""
    settings = get_settings()
    extra_body: dict = {}
    if settings.llm_disable_thinking:
        # DeepSeek-style thinking models: strip hidden reasoning, which is both
        # slower and less accurate on structured tool arguments.
        extra_body["thinking"] = {"type": "disabled"}
    return OpenAIBackend(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key or "EMPTY",
        timeout=settings.llm_request_timeout,
        extra_body=extra_body,
        stream=True,
    )

