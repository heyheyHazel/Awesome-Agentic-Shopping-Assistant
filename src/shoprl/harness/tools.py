"""Tool registry: JSON-schema declarations bound to deterministic handlers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from shoprl.harness.types import ToolCall

Handler = Callable[[dict[str, Any]], str]


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Handler

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """Name-indexed tools with a single entry point that never raises.

    A failing tool becomes an error observation instead of an exception: the
    model gets to correct itself, and a broken rollout is still trainable
    (negatively) rather than lost.
    """

    def __init__(self, specs: Iterable[ToolSpec] = ()):
        self._specs: dict[str, ToolSpec] = {spec.name: spec for spec in specs}

    def add(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: Handler,
    ) -> ToolSpec:
        spec = ToolSpec(name=name, description=description, parameters=parameters, handler=handler)
        self._specs[name] = spec
        return spec

    def schemas(self) -> list[dict[str, Any]]:
        return [spec.schema() for spec in self._specs.values()]

    @property
    def names(self) -> list[str]:
        return list(self._specs)

    def __contains__(self, name: object) -> bool:
        return name in self._specs

    def __getitem__(self, name: str) -> ToolSpec:
        return self._specs[name]

    def invoke(self, call: ToolCall) -> tuple[str, bool]:
        """Run one call, returning ``(observation, ok)``."""
        spec = self._specs.get(call.name)
        if spec is None:
            return f"Unknown tool `{call.name}`. Available tools: {', '.join(self._specs)}.", False
        try:
            return str(spec.handler(call.arguments or {})), True
        except Exception as exc:  # noqa: BLE001 - surfaced to the model as text
            return f"{call.name} failed: {type(exc).__name__}: {exc}", False

