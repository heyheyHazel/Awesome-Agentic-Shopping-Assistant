"""The channel tools use to push UI payloads onto the response stream.

Tools are plain functions with no reference to the transport, so the sink is a
context variable that the agent binds for the duration of one turn. With nothing
bound — a unit test calling a tool directly, or a script — events are dropped.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

Sink = Callable[[dict[str, Any]], None]

_sink: ContextVar[Sink | None] = ContextVar("event_sink", default=None)


@contextmanager
def bind(sink: Sink) -> Iterator[None]:
    """Route every event emitted inside the block to ``sink``."""
    token = _sink.set(sink)
    try:
        yield
    finally:
        _sink.reset(token)


def emit(event: dict[str, Any]) -> None:
    """Send one event to the bound sink, if there is one."""
    sink = _sink.get()
    if sink is not None:
        sink(event)


def tool_event(tool: str, status: str, message: str = "") -> None:
    """Report that a tool started or finished, for the tool panel in the UI."""
    emit({"type": "tool", "tool": tool, "status": status, "message": message})

