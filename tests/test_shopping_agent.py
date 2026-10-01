"""The serving loop: event protocol, answer selection, and conversation state."""

from __future__ import annotations

import asyncio

import pytest

from shoprl.harness.backends import ScriptedBackend
from shoprl.harness.types import ModelResponse, ToolCall
from shopping_assistant.agent.shopping_agent import ShoppingAgent


def call(name: str, **arguments) -> ToolCall:
    return ToolCall(id=f"{name}-1", name=name, arguments=arguments)


async def collect(agent: ShoppingAgent, message: str, thread: str = "t1") -> list[dict]:
    return [
        event
        async for event in agent.astream(message, user_id="U001", language="en", thread_id=thread)
    ]


def types_of(events: list[dict]) -> list[str]:
    return [event["type"] for event in events]


def test_a_recommendation_turn_emits_cards_and_the_reply():
    backend = ScriptedBackend(
        [
            ModelResponse(tool_calls=[call("search_catalog", query="香薰")]),
            ModelResponse(tool_calls=[call("present_recommendation", product_ids=["P001"])]),
            ModelResponse(text="Here is the one I would pick."),
        ],
        repeat_last=False,
    )
    events = asyncio.run(collect(ShoppingAgent(backend), "recommend a diffuser"))

    assert types_of(events)[0] == "tool"
    assert events[0]["tool"] == "assistant" and events[0]["status"] == "running"
    assert "products" in types_of(events)
    assert "".join(e["content"] for e in events if e["type"] == "token") == (
        "Here is the one I would pick."
    )
    assert events[-1] == {"type": "tool", "tool": "assistant", "status": "done", "message": ""}


def test_tool_calls_are_announced_before_they_run():
    backend = ScriptedBackend(
        [
            ModelResponse(tool_calls=[call("get_shopper_profile", user_id="U001")]),
            ModelResponse(text="done"),
        ],
        repeat_last=False,
    )
    events = asyncio.run(collect(ShoppingAgent(backend), "who am I"))
    statuses = [(e.get("tool"), e.get("status")) for e in events if e["type"] == "tool"]

    assert ("get_shopper_profile", "running") in statuses
    assert ("get_shopper_profile", "done") in statuses
    assert statuses.index(("get_shopper_profile", "running")) < statuses.index(
        ("get_shopper_profile", "done")
    )


def test_a_preamble_before_a_tool_call_is_not_shown_to_the_shopper():
    """The model's working notes must not reach the transcript the user reads."""
    backend = ScriptedBackend(
        [
            ModelResponse(text="Let me look that up.", tool_calls=[call("search_catalog", query="香薰")]),
            ModelResponse(text="The answer."),
        ],
        repeat_last=False,
    )
    events = asyncio.run(collect(ShoppingAgent(backend), "recommend a diffuser"))
    reply = "".join(e["content"] for e in events if e["type"] == "token")

    assert reply == "The answer."
    assert "look that up" not in reply


def test_a_follow_up_turn_reuses_the_conversation():
    backend = ScriptedBackend(
        [ModelResponse(tool_calls=[call("search_catalog", query="香薰")]), ModelResponse(text="ok")],
        repeat_last=False,
    )
    agent = ShoppingAgent(backend)
    asyncio.run(collect(agent, "recommend a diffuser"))
    asyncio.run(collect(agent, "and something cheaper"))

    second_turn = backend.calls[-1]
    assert any("recommend a diffuser" in message.content for message in second_turn)


def test_each_conversation_keeps_its_own_state():
    backend = ScriptedBackend([ModelResponse(tool_calls=[call("search_catalog", query="x")]), ModelResponse(text="ok")], repeat_last=False)
    agent = ShoppingAgent(backend)
    asyncio.run(collect(agent, "first thread", thread="a"))
    asyncio.run(collect(agent, "second thread", thread="b"))

    assert len(agent.session("a").messages) > 0
    assert all("second thread" not in m.content for m in agent.session("a").messages)


def test_a_failing_backend_surfaces_instead_of_hanging():
    class Broken:
        def complete(self, *_args, **_kwargs):
            raise RuntimeError("upstream is down")

    with pytest.raises(RuntimeError, match="upstream is down"):
        asyncio.run(collect(ShoppingAgent(Broken()), "hello"))

