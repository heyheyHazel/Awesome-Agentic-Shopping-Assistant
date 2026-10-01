"""Tool-call parsing for open-weight chat models.

Every mainstream open model wraps a call in ``<tool_call>`` with a JSON body, and
the field names differ only slightly, so one tolerant parser covers them. A parse
that fails returns no calls, which ends the episode rather than inventing an
action the model never asked for.
"""

from __future__ import annotations

import json
import re
from typing import Any

from shoprl.harness.types import ToolCall

_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FENCED = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _coerce(raw: Any) -> ToolCall | None:
    if not isinstance(raw, dict):
        return None
    function = raw.get("function") if isinstance(raw.get("function"), dict) else raw
    name = function.get("name") or raw.get("tool_name") or raw.get("tool")
    arguments = function.get("arguments", raw.get("arguments", raw.get("parameters")))
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"action": arguments}
    if not name:
        return None
    return ToolCall(id="", name=str(name), arguments=arguments if isinstance(arguments, dict) else {})


def parse_tool_calls(text: str) -> list[ToolCall]:
    """Extract tool calls from a model turn, newest formats first."""
    calls: list[ToolCall] = []
    for block in _BLOCK.findall(text or ""):
        try:
            payload = json.loads(block)
        except json.JSONDecodeError:
            payload = None
            for fenced in _FENCED.findall(block):
                try:
                    payload = json.loads(fenced)
                    break
                except json.JSONDecodeError:
                    continue
            if payload is None:
                continue
        for item in payload if isinstance(payload, list) else [payload]:
            call = _coerce(item)
            if call is not None:
                calls.append(call)
    for index, call in enumerate(calls):
        call.id = call.id or f"call_{index}"
    return calls


def strip_tool_calls(text: str) -> str:
    """The reply with any call blocks removed, for logging and final answers."""
    return _BLOCK.sub("", text or "").strip()

