"""Shared ChatOpenAI factory honouring provider-specific options."""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from config import get_settings


def build_llm(temperature: float, max_tokens: int) -> ChatOpenAI:
    """Create a ChatOpenAI client configured from settings.

    Providers whose thinking models slow down or reject structured output can
    opt into `ECOM_LLM_DISABLE_THINKING=true`, which strips hidden reasoning
    from every call.
    """
    settings = get_settings()
    extra: dict = {}
    if settings.llm_disable_thinking:
        extra["extra_body"] = {"thinking": {"type": "disabled"}}
    return ChatOpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        temperature=temperature,
        max_tokens=max_tokens,  # type: ignore[reportCallIssue]
        timeout=settings.llm_request_timeout,
        **extra,
    )
