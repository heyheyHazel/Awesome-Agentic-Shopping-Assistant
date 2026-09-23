"""Provider-friendly structured output for OpenAI-compatible chat models.

Uses JSON mode plus an injected JSON schema instead of forced tool calls or
json_schema response formats, which some providers (e.g. DeepSeek thinking
models) reject with HTTP 400.
"""

from __future__ import annotations

import json
from typing import Any, TypeVar

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def text_of(content: Any) -> str:
    """Normalise LLM content (str or content-block list) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return ""


class JsonStructured:
    """Calls the model in JSON mode and validates the reply against a schema."""

    def __init__(self, llm: ChatOpenAI, schema: type[T]):
        self.llm = llm.bind(response_format={"type": "json_object"})
        self.schema = schema
        self._instruction = SystemMessage(
            content=(
                "Respond with a single JSON object that matches this JSON schema exactly "
                "(additional text is not allowed):\n"
                + json.dumps(schema.model_json_schema())
            )
        )

    async def ainvoke(self, messages: list[BaseMessage]) -> T:
        response = await self.llm.ainvoke([self._instruction, *messages])
        return self.schema.model_validate_json(text_of(response.content))
