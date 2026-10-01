"""Harness behaviour: loop termination, context pruning, tool failures, memory."""

from __future__ import annotations

from shoprl.harness.context import PRUNED_TOOL_RESULT, ContextPolicy
from shoprl.harness.loop import TERMINAL_MARKER, AgentLoop
from shoprl.harness.memory import Memory
from shoprl.harness.tools import ToolRegistry
from shoprl.harness.types import Message, ModelResponse, ToolCall
from shoprl.harness.backends import ScriptedBackend

ECHO_SCHEMA = {"type": "object", "properties": {"value": {"type": "string"}}}


def build_loop(backend, **kwargs) -> AgentLoop:
    registry = ToolRegistry()
    registry.add("shop_reset", "reset", {"type": "object", "properties": {}}, lambda _a: "Instruction: buy a cup")
    registry.add("shop_act", "act", ECHO_SCHEMA, lambda a: f"page({a.get('value')})")
    return AgentLoop(backend, registry, **kwargs)


def call(name: str, **arguments) -> ToolCall:
    return ToolCall(id=f"{name}-1", name=name, arguments=arguments)


def test_loop_records_every_step_and_the_context_each_one_saw():
    backend = ScriptedBackend(
        [
            ModelResponse(tool_calls=[call("shop_reset")]),
            ModelResponse(tool_calls=[call("shop_act", value="search[a]")]),
            ModelResponse(text="finished"),
        ],
        repeat_last=False,
    )
    trajectory = build_loop(backend, max_turns=5).run("do the task", task_id=7)

    assert trajectory.task_id == 7
    assert trajectory.model_turns == 3
    assert [len(step.context) for step in trajectory.steps] == [2, 4, 6]
    assert trajectory.messages[-1].content == "finished"


def test_loop_stops_as_soon_as_a_tool_reports_the_task_is_over():
    backend = ScriptedBackend(
        [
            ModelResponse(tool_calls=[call("shop_reset")]),
            ModelResponse(tool_calls=[call("shop_act", value="buy")]),
        ]
    )
    loop = build_loop(backend)
    loop.tools["shop_act"].handler = lambda _a: f"{TERMINAL_MARKER}reward=1.0"

    trajectory = loop.run("do the task", task_id=1)
    assert trajectory.termination == "done"
    assert trajectory.model_turns == 2


def test_loop_reports_a_dead_backend_instead_of_raising():
    class Broken:
        def complete(self, *_args, **_kwargs):
            raise RuntimeError("connection reset")

    trajectory = build_loop(Broken()).run("do the task")
    assert trajectory.termination == "backend_error"
    assert "connection reset" in trajectory.error


def test_unknown_tools_become_observations_not_exceptions():
    backend = ScriptedBackend(
        [ModelResponse(tool_calls=[call("nope")]), ModelResponse(text="ok")], repeat_last=False
    )
    trajectory = build_loop(backend, max_turns=4).run("do the task")
    observation = trajectory.steps[0].observations[0].content
    assert "Unknown tool" in observation
    assert "shop_act" in observation


def test_context_policy_prunes_old_tool_results_but_keeps_the_pairing():
    messages = [Message(role="system", content="s"), Message(role="user", content="u")]
    for index in range(4):
        messages.append(Message(role="assistant", content="", tool_calls=[call("shop_act", value=str(index))]))
        messages.append(Message(role="tool", content=f"page {index}", tool_call_id=f"shop_act-1"))

    view = ContextPolicy(keep_recent_tool_results=2).apply(messages)

    assert len(view) == len(messages)
    assert [message.role for message in view] == [message.role for message in messages]
    results = [message.content for message in view if message.role == "tool"]
    assert results == [PRUNED_TOOL_RESULT, PRUNED_TOOL_RESULT, "page 2", "page 3"]
    # Every tool result still points at the call that produced it.
    assert all(
        view[index].tool_call_id
        for index, message in enumerate(view)
        if message.role == "tool"
    )


def test_context_policy_drops_whole_oldest_steps_under_a_token_budget():
    messages = [Message(role="system", content="system"), Message(role="user", content="task")]
    for index in range(6):
        messages.append(Message(role="assistant", content="", tool_calls=[call("shop_act", value=str(index))]))
        messages.append(Message(role="tool", content="x" * 400, tool_call_id="shop_act-1"))

    view = ContextPolicy(keep_recent_tool_results=99, max_prompt_tokens=200).apply(messages)

    assert view[0].content == "system"
    assert view[1].content == "task"
    assert len(view) < len(messages)
    # Dropping is step-aligned, and the newest step always survives.
    assert view[2].role == "assistant"
    assert view[-1].content == "x" * 400
    assert sum(1 for message in view if message.role == "assistant") == 1


def test_memory_is_bounded_and_deduplicated_by_key():
    memory = Memory(max_notes=3)
    memory.remember("budget is 500", key="budget")
    memory.remember("budget is 300", key="budget")
    memory.remember("colour must be black", key="colour")

    rendered = memory.render()
    assert "budget is 300" in rendered and "colour must be black" in rendered
    assert "budget is 500" not in rendered
    assert len(memory.notes) == 2

    memory.remember("fourth", key="fourth")
    memory.remember("fifth", key="fifth")
    assert len(memory.notes) == 3
    assert "budget is 300" not in memory.render()
