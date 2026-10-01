"""The shopping agent: one tool-calling loop handling every turn.

This is the same loop the training pipeline runs. The only thing that differs
between serving and training is where the events go: here they are pushed onto
the SSE response, while a rollout keeps the recorded trajectory instead. That is
deliberate — a demo that runs a different agent from the one being trained proves
nothing about either.

There is exactly one LLM loop in the system. Intent routing, query rewriting,
picking products and writing the reply all happen inside it, which is why the
model needs tools only for what it genuinely cannot know: the catalog, the
shopper's history and live stock.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from shoprl.harness.context import ContextPolicy
from shoprl.harness.loop import AgentLoop
from shoprl.harness.memory import Memory
from shoprl.harness.types import Message, Trajectory

from shopping_assistant.agent.prompts import catalog_language_rule, language_directive
from shopping_assistant.agent.tools import build_tool_registry
from shopping_assistant.catalog import PRODUCTS
from shopping_assistant.services.events import bind
from shopping_assistant.services.llm import build_backend
from shopping_assistant.settings import get_settings

CATEGORIES = sorted({product.category for product in PRODUCTS})

SYSTEM_PROMPT = f"""You are the shopping assistant for an online store, and you handle the whole
turn yourself: work out what the shopper wants, look up whatever you need, and answer.

Catalog categories: {', '.join(CATEGORIES)}

Tools
- get_shopper_profile: the shopper's segment, category preferences and usual budget.
- search_catalog: candidate products. Budget and category are hard filters and sold-out
  items are never returned. Call it whenever you need products to talk about.
- check_inventory: live stock for specific products. Call it for anything you plan to recommend.
- present_recommendation: shows the shopper the product cards for your final picks.

How to work
1. If the shopper states a budget or a category, pass it to search_catalog. A stated budget
   is a limit you must never exceed, even if nothing cheaper is available. Search once with
   good parameters rather than repeatedly: search again only if the results were unusable.
2. `query` is the request in your own words; `keywords` are 1-4 short catalog terms for exact
   matching, such as brands, materials or model numbers.
3. Look up the profile when personalisation helps. Do not force preferences that are
   irrelevant to the request.
4. Check stock, then call present_recommendation with the 2-3 products you want to show. The
   cards are capped at 3 and the tool reports which ones were actually shown — write about
   exactly those and no others.
5. Finish with two or three sentences: name what you picked and why it fits this shopper.
   Concrete and plain — no price list, no hype. Never say "four" when three cards are shown.
6. If nothing in the catalog fits, say so plainly and do not present anything.
7. Never narrate what you are about to do. Call tools silently; the shopper only ever sees
   your final reply.

{catalog_language_rule(CATEGORIES)}"""

TOKEN_CHUNK = 6


def _chunks(text: str) -> list[str]:
    """Split the reply so the UI keeps its typewriter pacing."""
    return [text[i : i + TOKEN_CHUNK] for i in range(0, len(text), TOKEN_CHUNK)]


@dataclass
class Session:
    """Per-conversation state: the transcript plus the notes the model pinned."""

    messages: list[Message] = field(default_factory=list)
    memory: Memory = field(default_factory=Memory)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class ShoppingAgent:
    """One tool-calling loop behind a single `astream` that yields SSE events."""

    def __init__(self, backend: Any = None):
        self.backend = backend
        self._sessions: dict[str, Session] = {}

    def session(self, thread_id: str) -> Session:
        return self._sessions.setdefault(thread_id, Session())

    def backend_or_default(self) -> Any:
        """Build the configured client once; the module stays importable without credentials."""
        if self.backend is None:
            self.backend = build_backend()
        return self.backend

    async def astream(
        self, message: str, *, user_id: str, language: str, thread_id: str
    ) -> AsyncIterator[dict[str, Any]]:
        """Run one turn, yielding the events the SSE endpoint forwards.

        Tools emit their own `tool`/`products`/`profile`/`inventory` events while
        they run. The reply text is emitted here, and only for the model message
        that turns out to be the answer: a tool-calling turn produces several
        model messages, and the ones that precede a tool call are working notes
        rather than something the shopper should read.
        """
        session = self.session(thread_id)
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def emit(event: dict[str, Any]) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, event)

        yield {"type": "tool", "tool": "assistant", "status": "running", "message": ""}
        async with session.lock:
            turn = asyncio.create_task(
                asyncio.to_thread(self._run_turn, session, message, user_id, language, emit)
            )
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield event
            await turn
        yield {"type": "tool", "tool": "assistant", "status": "done", "message": ""}

    # ── internals ─────────────────────────────────────────────────────

    def _run_turn(
        self,
        session: Session,
        message: str,
        user_id: str,
        language: str,
        emit,
    ) -> None:
        """Blocking turn body; runs on a worker thread so the event loop stays free."""
        settings = get_settings()
        agent = AgentLoop(
            self.backend_or_default(),
            build_tool_registry(),
            system_prompt=self._system_prompt(user_id, language),
            context_policy=ContextPolicy(keep_recent_tool_results=3),
            memory=session.memory,
            max_turns=settings.max_turns,
            max_tool_calls=settings.max_tool_calls,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            emit=emit,
        )
        try:
            with bind(emit):
                trajectory = agent.run(message, history=session.messages)
            if trajectory.termination == "backend_error":
                # The harness never raises: a failed rollout is still recorded, and
                # training wants the failure. Serving has to turn it into an error.
                raise RuntimeError(trajectory.error or "the model call failed")
            session.messages = [item for item in trajectory.messages if item.role != "system"]
            for piece in _chunks(_reply_of(trajectory)):
                emit({"type": "token", "content": piece})
        finally:
            # The sentinel always goes last: it closes the stream, and it must be
            # sent even when the turn raised, or the caller waits forever.
            emit(None)

    def _system_prompt(self, user_id: str, language: str) -> str:
        return (
            f"{SYSTEM_PROMPT}\n"
            f"The current shopper's user_id is {user_id or 'unknown'}.\n"
            f"{language_directive(language)}"
        )


def _reply_of(trajectory: Trajectory) -> str:
    """The final answer, or "" when the model stopped without writing one."""
    if not trajectory.steps:
        return ""
    assistant = trajectory.steps[-1].assistant
    return "" if assistant.tool_calls else assistant.content
