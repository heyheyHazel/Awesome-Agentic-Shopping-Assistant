"""End-to-end graph behaviour with stubbed LLMs (no API key needed)."""

from langchain_core.messages import AIMessage, HumanMessage

from models.schemas import (
    CopyItem,
    CopySet,
    ProductRanking,
    SearchParams,
    SupervisorPlan,
)
from orchestrator import graph as graph_module
from orchestrator.graph import build_graph

USER_MESSAGE = "I'm looking for running shoes for daily training under $120."

FULL_PLAN = SupervisorPlan(
    intent="product_search",
    reply="Got it! Looking for running shoes under $120.",
    search=SearchParams(keywords=["running shoes"], max_price=120),
    agents=["profile", "recall", "rerank", "inventory", "copy"],
)


class StubStructured:
    """Stands in for llm.with_structured_output(...)."""

    def __init__(self, result):
        self.result = result
        self.calls: list = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return self.result


class BrokenStructured:
    async def ainvoke(self, _messages):
        raise RuntimeError("llm unavailable")


class StubReplyLLM:
    async def astream(self, _messages):
        for token in ["Great ", "choice!"]:
            yield AIMessage(content=token)


class StubChatAgent:
    async def astream(self, _messages, _user_id, _language="en"):
        yield "Hello "
        yield "there!"


def stub_supervisor(monkeypatch, plan: SupervisorPlan):
    monkeypatch.setattr(graph_module.supervisor_agent, "planner", StubStructured(plan))
    monkeypatch.setattr(graph_module, "reply_llm", StubReplyLLM())
    monkeypatch.setattr(graph_module, "chat_agent", StubChatAgent())


async def test_product_search_turn_produces_ranked_products(monkeypatch):
    stub_supervisor(monkeypatch, FULL_PLAN)
    monkeypatch.setattr(
        graph_module.rec_agent, "ranker",
        StubStructured(ProductRanking(product_ids=["P003", "P001", "P002"])),
    )
    monkeypatch.setattr(
        graph_module.copy_agent, "writer",
        StubStructured(CopySet(items=[CopyItem(product_id="P003", text="Light and stable.")])),
    )

    graph = build_graph()
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=USER_MESSAGE)], "query": USER_MESSAGE, "user_id": "U001"},
        config={"configurable": {"thread_id": "search-1"}},
    )

    assert [p.product_id for p in result["final_products"]] == ["P003", "P001", "P002"]
    assert result["profile"].segment == "Champions"
    assert result["copies"][0].text == "Light and stable."
    assert result["reply"] == "Great choice!"


async def test_plan_can_skip_profile_and_copy(monkeypatch):
    lean_plan = FULL_PLAN.model_copy(update={"agents": ["recall", "rerank", "inventory"]})
    stub_supervisor(monkeypatch, lean_plan)
    monkeypatch.setattr(
        graph_module.rec_agent, "ranker",
        StubStructured(ProductRanking(product_ids=["P001", "P002", "P003"])),
    )

    graph = build_graph()
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=USER_MESSAGE)], "query": USER_MESSAGE, "user_id": "U001"},
        config={"configurable": {"thread_id": "lean-1"}},
    )

    assert len(result["final_products"]) == 3
    assert result["profile"] is None
    assert result["copies"] == []


async def test_general_turn_uses_chat_agent(monkeypatch):
    general = SupervisorPlan(intent="general", reply="Hi!", agents=[])
    stub_supervisor(monkeypatch, general)

    graph = build_graph()
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="hello")], "query": "hello", "user_id": "U002"},
        config={"configurable": {"thread_id": "general-1"}},
    )

    assert result["reply"] == "Hello there!"
    assert getattr(result["messages"][-1], "content", "") == "Hello there!"


async def test_rerank_failure_degrades_to_recall_order(monkeypatch):
    stub_supervisor(monkeypatch, FULL_PLAN)
    monkeypatch.setattr(graph_module.rec_agent, "ranker", BrokenStructured())

    graph = build_graph()
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=USER_MESSAGE)], "query": USER_MESSAGE, "user_id": "U001"},
        config={"configurable": {"thread_id": "degrade-1"}},
    )

    assert result["ranked"] == []
    assert len(result["final_products"]) == 3
    assert result["reply"] == "Great choice!"


async def test_language_directive_reaches_the_supervisor(monkeypatch):
    planner = StubStructured(SupervisorPlan(intent="general", reply="你好！", agents=[]))
    monkeypatch.setattr(graph_module.supervisor_agent, "planner", planner)
    monkeypatch.setattr(graph_module, "chat_agent", StubChatAgent())

    graph = build_graph()
    await graph.ainvoke(
        {"messages": [HumanMessage(content="你好")], "query": "你好", "user_id": "U001", "language": "zh"},
        config={"configurable": {"thread_id": "lang-1"}},
    )

    system_prompt = planner.calls[0][0].content
    assert "Simplified Chinese" in system_prompt
    assert "English catalog terms" in system_prompt


async def test_checkpointer_keeps_conversation_history(monkeypatch):
    stub_supervisor(monkeypatch, FULL_PLAN)
    monkeypatch.setattr(
        graph_module.rec_agent, "ranker",
        StubStructured(ProductRanking(product_ids=["P001", "P002", "P003"])),
    )
    monkeypatch.setattr(
        graph_module.copy_agent, "writer",
        StubStructured(CopySet(items=[CopyItem(product_id="P001", text="A solid pick.")])),
    )

    graph = build_graph()
    config = {"configurable": {"thread_id": "memory-1"}}
    await graph.ainvoke(
        {"messages": [HumanMessage(content=USER_MESSAGE)], "query": USER_MESSAGE, "user_id": "U001"},
        config=config,
    )
    second = await graph.ainvoke(
        {"messages": [HumanMessage(content="what about cheaper ones?")], "query": "what about cheaper ones?", "user_id": "U001"},
        config=config,
    )

    history = [m.content for m in second["messages"]]
    assert any("under $120" in str(c) for c in history)
    assert len(second["messages"]) == 4
