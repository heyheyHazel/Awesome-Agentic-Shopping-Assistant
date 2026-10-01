"""Binds one live shop episode to harness tools.

Two tools, mirroring the reference harness: ``shop_reset`` starts the assigned
task and ``shop_act`` sends one native action. The environment owns the task, so
the prompt never contains the instruction; that matches upstream and keeps a
trained model able to run against the real environment unchanged.
"""

from __future__ import annotations

from typing import Any, Callable

from shoprl.env.actions import END_BUTTON
from shoprl.env.persona import summarise as summarise_persona
from shoprl.harness.memory import Memory
from shoprl.harness.tools import ToolRegistry
from shoprl.harness.loop import TERMINAL_MARKER

ACT_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "minLength": 1,
            "description": "One action, exactly as `search[keywords]` or `click[value]`.",
        }
    },
    "required": ["action"],
    "additionalProperties": False,
}

RESET_DESCRIPTION = (
    "Start the assigned shopping task. Call exactly once before shop_act. "
    "Returns the shopper's instruction and the first page."
)

ACT_DESCRIPTION = (
    "Send one action to the store and get the resulting page. "
    "`search[keywords]` searches the catalogue. "
    "`click[value]` opens a product, selects an option, pages, or buys. "
    "Only values listed in 可点击的按钮 of the last observation are accepted, "
    f"and `click[{END_BUTTON}]` finishes the task."
)


class ShopToolkit:
    """One episode's tools plus the verdict the environment produced for it."""

    def __init__(
        self,
        env: Any,
        task_id: int,
        session_id: str,
        *,
        memory: Memory | None = None,
        on_close: Callable[[], None] | None = None,
        show_persona: bool = False,
    ):
        self.env = env
        self.task_id = task_id
        self.session_id = session_id
        self.memory = memory
        self._on_close = on_close
        self.show_persona = show_persona

        self._reset_done = False
        self._terminal = False
        self._last: dict[str, Any] = {}
        self._turn = 0

    # ── tools ─────────────────────────────────────────────────────────

    def registry(self) -> ToolRegistry:
        registry = ToolRegistry()
        registry.add(
            "shop_reset",
            RESET_DESCRIPTION,
            {"type": "object", "properties": {}, "additionalProperties": False},
            self._reset,
        )
        registry.add("shop_act", ACT_DESCRIPTION, ACT_SCHEMA, self._act)
        if self.memory is not None:
            registry.add("save_note", NOTE_DESCRIPTION, NOTE_SCHEMA, self._note)
        return registry

    def outcome(self) -> dict[str, Any]:
        return {
            "reward": float(self._last.get("reward") or 0.0),
            "reward_detail": dict(self._last.get("reward_detail") or {}),
            "purchase": self._last.get("purchase") or None,
            "goal": self._last.get("goal") or None,
        }

    def close(self) -> None:
        """Return the environment. The pool owns the slot, so it decides how."""
        if self._on_close is not None:
            self._on_close()
        else:
            self.env.release(self.session_id)

    @property
    def finished(self) -> bool:
        return self._terminal

    # ── handlers ──────────────────────────────────────────────────────

    def _reset(self, _arguments: dict[str, Any]) -> str:
        if self._reset_done:
            return "shop_reset was already called; continue with shop_act."
        result = self.env.reset(self.task_id, self.session_id)
        self._reset_done = True
        instruction = result.get("instruction") or result.get("message") or ""
        persona = summarise_persona(result.get("user_persona")) if self.show_persona else ""
        return "\n\n".join(part for part in [f"Instruction: {instruction}", persona] if part)

    def _act(self, arguments: dict[str, Any]) -> str:
        if self._terminal:
            return f"{TERMINAL_MARKER}the task already ended; take no further action."
        if not self._reset_done:
            return "Call shop_reset before shop_act."

        action = str(arguments.get("action") or "").strip()
        if not action:
            return "shop_act needs a non-empty `action`."
        self._turn += 1

        result = self.env.interact(action, self.session_id)
        if result.get("done"):
            self._terminal = True
            self._last = result
            return (
                f"{TERMINAL_MARKER}reward={result.get('reward', 0.0)}. "
                f"{result.get('instruction', '')}"
            )
        if result.get("over"):
            self._terminal = True
            self._last = {**result, "reward": 0.0}
            return f"{TERMINAL_MARKER}the store ended this session without a purchase."
        if not result.get("changed", True):
            return "That action is not available on this page. Nothing changed.\n" + str(
                result.get("instruction", "")
            )
        return str(result.get("instruction", ""))

    def _note(self, arguments: dict[str, Any]) -> str:
        assert self.memory is not None
        self.memory.remember(str(arguments.get("content") or ""), turn=self._turn)
        return "Noted."


NOTE_DESCRIPTION = (
    "Save a short note that stays visible for the rest of this session, for example a "
    "hard constraint from the shopper. Use it for requirements you must not lose."
)
NOTE_SCHEMA = {
    "type": "object",
    "properties": {"content": {"type": "string", "minLength": 1}},
    "required": ["content"],
    "additionalProperties": False,
}
