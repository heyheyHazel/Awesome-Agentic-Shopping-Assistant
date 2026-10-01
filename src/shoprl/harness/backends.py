"""Model backends: anything that turns a message list into one assistant turn.

``OpenAIBackend`` speaks the Chat Completions protocol, so it serves both a
hosted teacher and a local vLLM/SGLang rollout server without a second code path.
Backends that can report sampled token ids and logprobs fill them in; backends
that cannot leave them empty and the training code falls back to re-tokenising.

``on_token`` is optional and only ever used for display: serving streams deltas to
the browser, training passes nothing and reads the finished response.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Protocol

import httpx

from shoprl.harness.types import Message, ModelResponse, ToolCall

TokenSink = Callable[[str], None]


class ModelBackend(Protocol):
    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: int | None = None,
        on_token: TokenSink | None = None,
    ) -> ModelResponse: ...


def _parse_tool_calls(raw: list[dict[str, Any]] | None) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for index, item in enumerate(raw or []):
        function = item.get("function") or {}
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError:
                arguments = {"_raw": arguments}
        calls.append(
            ToolCall(
                id=str(item.get("id") or f"call_{index}"),
                name=str(function.get("name") or ""),
                arguments=arguments if isinstance(arguments, dict) else {},
            )
        )
    return calls


class OpenAIBackend:
    """Chat Completions client for hosted teachers and local rollout servers."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:8000/v1",
        api_key: str = "EMPTY",
        timeout: float = 300.0,
        extra_body: dict[str, Any] | None = None,
        stream: bool = False,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.extra_body = extra_body or {}
        self.stream = stream
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: int | None = None,
        on_token: TokenSink | None = None,
    ) -> ModelResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [message.as_dict() for message in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if seed is not None:
            payload["seed"] = seed
        payload.update(self.extra_body)

        if self.stream and on_token is not None:
            return self._streamed(payload, on_token)

        response = self._client.post("/chat/completions", json=payload)
        response.raise_for_status()
        choice = response.json()["choices"][0]
        message = choice.get("message") or {}
        return ModelResponse(
            text=message.get("content") or "",
            tool_calls=_parse_tool_calls(message.get("tool_calls")),
            finish_reason=str(choice.get("finish_reason") or "stop"),
        )

    def _streamed(self, payload: dict[str, Any], on_token: TokenSink) -> ModelResponse:
        """Streamed variant; falls back to the buffered shape on the way out."""
        payload = {**payload, "stream": True}
        text_parts: list[str] = []
        partial: dict[int, dict[str, Any]] = {}
        finish_reason = "stop"

        with self._client.stream("POST", "/chat/completions", json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                try:
                    chunk = json.loads(body)
                except json.JSONDecodeError:
                    continue
                choice = (chunk.get("choices") or [{}])[0]
                finish_reason = str(choice.get("finish_reason") or finish_reason)
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    text_parts.append(delta["content"])
                    on_token(delta["content"])
                for call in delta.get("tool_calls") or []:
                    slot = partial.setdefault(int(call.get("index", 0)), {"id": "", "name": "", "arguments": ""})
                    if call.get("id"):
                        slot["id"] = call["id"]
                    function = call.get("function") or {}
                    slot["name"] += function.get("name") or ""
                    slot["arguments"] += function.get("arguments") or ""

        calls: list[ToolCall] = []
        for index in sorted(partial):
            slot = partial[index]
            raw = slot["arguments"]
            try:
                arguments = json.loads(raw) if raw.strip() else {}
            except json.JSONDecodeError:
                arguments = {"_raw": raw}
            calls.append(
                ToolCall(id=slot["id"] or f"call_{index}", name=slot["name"], arguments=arguments)
            )
        return ModelResponse(
            text="".join(text_parts), tool_calls=calls, finish_reason=finish_reason
        )

    def close(self) -> None:
        self._client.close()


class ScriptedBackend:
    """Replays canned responses; the default backend for tests and dry runs."""

    def __init__(self, responses: list[ModelResponse], *, repeat_last: bool = True):
        self._responses = list(responses)
        self._repeat_last = repeat_last
        self.calls: list[list[Message]] = []

    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: int | None = None,
        on_token: TokenSink | None = None,
    ) -> ModelResponse:
        self.calls.append(list(messages))
        if not self._responses:
            return ModelResponse(text="done", finish_reason="stop")
        if len(self._responses) == 1 and self._repeat_last:
            response = self._responses[0]
        else:
            response = self._responses.pop(0)
        if on_token is not None and response.text:
            on_token(response.text)
        return response
