"""End-to-end graph behaviour with stubbed LLMs (no API key needed)."""

from langchain_core.messages import HumanMessage

from models.schemas import CopyItem, ProductRanking, SearchParams, SupervisorPlan
from orchestrator import graph as graph_module
from orchestrator.graph import build_graph

USER_MESSAGE = "I'm looking for running shoes for daily training under ¥800."

FULL_PLAN = SupervisorPlan(
    intent="product_search",
    reply="Got it! Looking for running shoes under ¥800.",
    search=SearchParams(keywords=["running shoes"], max_price=800),
)


class StubStructured:
    """Stands in for JsonStructured."""

    def __init__(self, result):
        self.result = result
        self.calls: list = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return self.result


class BrokenStructured:
    async def ainvoke(self, _messages):
        raise RuntimeError("llm unavailable")


class StubChatAgent:
    async def astream(self, _messages, _user_id, _language="en"):
        yield "Hello "
        yield "there!"


def ranking(*ids: str, text: str = "A solid pick.") -> ProductRanking:
    return ProductRanking(
        product_ids=list(ids),
        pitches=[CopyItem(product_id=pid, text=text) for pid in ids],
    )


def stub_llms(monkeypatch, plan: SupervisorPlan, rank: ProductRanking):
    """Wire the two LLM calls a shopping turn makes and hand back their stubs."""
    planner, ranker = StubStructured(plan), StubStructured(rank)
    monkeypatch.setattr(graph_module.supervisor_agent, "planner", planner)
    monkeypatch.setattr(graph_module.rec_agent, "ranker", ranker)
    monkeypatch.setattr(graph_module, "chat_agent", StubChatAgent())
    return planner, ranker


async def run_turn(thread_id: str, message: str = USER_MESSAGE):
    return await build_graph().ainvoke(
        {"messages": [HumanMessage(content=message)], "query": message, "user_id": "U001"},
        config={"configurable": {"thread_id": thread_id}},
    )


async def test_product_search_turn_produces_ranked_products(monkeypatch):
    stub_llms(monkeypatch, FULL_PLAN, ranking("P003", "P001", "P002", text="Light and stable."))

    result = await run_turn("search-1")

    assert [p.product_id for p in result["final_products"]] == ["P003", "P001", "P002"]
    assert result["profile"].segment == "Champions"
    assert result["pitches"][0].text == "Light and stable."
    assert result["reply"] == "Light and stable.\nLight and stable.\nLight and stable."


async def test_a_shopping_turn_costs_exactly_two_llm_calls(monkeypatch):
    """Guards the pipeline against creeping back to one call per agent."""
    planner, ranker = stub_llms(monkeypatch, FULL_PLAN, ranking("P001", "P002", "P003"))

    await run_turn("budget-1")

    assert len(planner.calls) == 1
    assert len(ranker.calls) == 1


async def test_out_of_stock_candidates_never_reach_the_ranking_call(monkeypatch):
    """P005 is the cheapest match but has no stock, so it must not be offered."""
    _, ranker = stub_llms(monkeypatch, FULL_PLAN, ranking("P001", "P002", "P003"))

    result = await run_turn("stock-1")

    assert "P005" not in ranker.calls[0][1].content
    assert "P005" not in [p.product_id for p in result["final_products"]]


async def test_no_matches_skips_the_ranking_call(monkeypatch):
    tight = FULL_PLAN.model_copy(
        update={"search": SearchParams(keywords=["running shoes"], max_price=1)}
    )
    _, ranker = stub_llms(monkeypatch, tight, ranking("P001"))

    result = await run_turn("empty-1")

    assert result["final_products"] == []
    assert result["reply"] == ""
    assert ranker.calls == []


async def test_general_turn_uses_chat_agent(monkeypatch):
    stub_llms(monkeypatch, SupervisorPlan(intent="general", reply="Hi!"), ranking("P001"))

    result = await run_turn("general-1", "hello")

    assert result["reply"] == "Hello there!"
    assert getattr(result["messages"][-1], "content", "") == "Hello there!"


async def test_ranking_failure_degrades_to_recall_order(monkeypatch):
    stub_llms(monkeypatch, FULL_PLAN, ranking("P001"))
    monkeypatch.setattr(graph_module.rec_agent, "ranker", BrokenStructured())

    result = await run_turn("degrade-1")

    # Recall order is keyword hits first, then rating: P002 6.7 > P001 6.6 > P003 6.4.
    assert [p.product_id for p in result["final_products"]] == ["P002", "P001", "P003"]


async def test_language_directive_reaches_the_supervisor(monkeypatch):
    planner, _ = stub_llms(
        monkeypatch, SupervisorPlan(intent="general", reply="你好！"), ranking("P001")
    )

    await build_graph().ainvoke(
        {"messages": [HumanMessage(content="你好")], "query": "你好", "user_id": "U001", "language": "zh"},
        config={"configurable": {"thread_id": "lang-1"}},
    )

    system_prompt = planner.calls[0][0].content
    assert "Simplified Chinese" in system_prompt
    assert "English catalog terms" in system_prompt


async def test_checkpointer_keeps_conversation_history(monkeypatch):
    stub_llms(monkeypatch, FULL_PLAN, ranking("P001", "P002", "P003"))

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
    assert any("under ¥800" in str(c) for c in history)
    assert len(second["messages"]) == 4
