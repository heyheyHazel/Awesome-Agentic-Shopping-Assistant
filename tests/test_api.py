"""HTTP layer."""

def test_meta_endpoint_reports_the_catalog_currency():
    from fastapi.testclient import TestClient

    from shopping_assistant.api.app import app

    body = TestClient(app).get("/api/v1/meta").json()
    assert body["currency"] == "CNY"
    assert body["currency_symbol"] == "¥"
    assert body["data_source"] in {"mock", "shopsimulator"}


def parse_sse(body: str) -> list[tuple[str, dict]]:
    import json

    events = []
    for block in body.split("\n\n"):
        if not block.strip():
            continue
        name = next(line[7:] for line in block.splitlines() if line.startswith("event: "))
        data = next(line[6:] for line in block.splitlines() if line.startswith("data: "))
        events.append((name, json.loads(data)))
    return events


def test_chat_streams_the_whole_protocol_without_an_llm(monkeypatch):
    """The serving loop, end to end, with a scripted model.

    This is the link between the demo and the training harness: the same
    ``AgentLoop`` produces these events, and the SSE protocol the frontend reads
    is unchanged by that.
    """
    from fastapi.testclient import TestClient

    from shoprl.harness.backends import ScriptedBackend
    from shoprl.harness.types import ModelResponse, ToolCall
    from shopping_assistant.agent.shopping_agent import ShoppingAgent
    from shopping_assistant.api import app as app_module

    def call(name, **arguments):
        return ToolCall(id=name, name=name, arguments=arguments)

    backend = ScriptedBackend(
        [
            ModelResponse(tool_calls=[call("search_catalog", query="香薰")]),
            ModelResponse(tool_calls=[call("present_recommendation", product_ids=["P001"])]),
            ModelResponse(text="推荐这一款。"),
        ],
        repeat_last=False,
    )
    monkeypatch.setattr(app_module, "_agent", ShoppingAgent(backend))

    with TestClient(app_module.app) as client:
        with client.stream(
            "POST",
            "/api/v1/chat",
            json={"user_id": "U001", "message": "推荐一个香薰", "language": "zh", "thread_id": "t-1"},
        ) as response:
            assert response.status_code == 200
            events = parse_sse("".join(response.iter_text()))

    names = [name for name, _ in events]
    assert names[0] == "session"
    assert "experiment" in names
    assert "products" in names
    assert "done" in names and names[-1] == "done"

    by_name = dict(events)
    assert by_name["session"]["thread_id"] == "t-1"
    assert [p["product_id"] for p in by_name["products"]["products"]] == ["P001"]

    reply = "".join(payload["content"] for name, payload in events if name == "token")
    assert reply == "推荐这一款。"

    tools = {(payload["tool"], payload["status"]) for name, payload in events if name == "tool"}
    assert ("search_catalog", "running") in tools
    assert ("present_recommendation", "done") in tools
    assert ("assistant", "done") in tools
    assert by_name["done"]["latency_ms"] >= 0

    # Exactly one running/done pair per tool: a duplicate would make the latency
    # panel report the gap between two of them instead of the tool's duration.
    from collections import Counter

    pairs = Counter(
        (payload["tool"], payload["status"]) for name, payload in events if name == "tool"
    )
    assert pairs[("search_catalog", "running")] == 1
    assert pairs[("search_catalog", "done")] == 1
    assert pairs[("present_recommendation", "running")] == 1
    assert set(by_name["done"]["timings"]) >= {"search_catalog", "present_recommendation"}


def test_chat_reports_a_model_failure_as_an_error_event(monkeypatch):
    from fastapi.testclient import TestClient

    from shopping_assistant.agent.shopping_agent import ShoppingAgent
    from shopping_assistant.api import app as app_module

    class Broken:
        def complete(self, *_args, **_kwargs):
            raise RuntimeError("upstream is down")

    monkeypatch.setattr(app_module, "_agent", ShoppingAgent(Broken()))

    with TestClient(app_module.app) as client:
        response = client.post(
            "/api/v1/chat", json={"user_id": "U001", "message": "hi", "language": "en"}
        )

    names = [name for name, _ in parse_sse(response.text)]
    assert "error" in names
    assert names[-1] == "error"


def test_the_serving_path_imports_without_an_agent_framework():
    """The demo runs the project's own loop, not a framework's.

    Checked in a subprocess with the framework packages blocked, because this
    machine has them installed and an accidental re-import would otherwise go
    unnoticed until a clean deployment.
    """
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import sys, os
        BLOCKED = {"langchain", "langchain_core", "langchain_openai", "langgraph"}

        class Blocker:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in BLOCKED:
                    raise ImportError(f"blocked: {name}")
                return None

        sys.meta_path.insert(0, Blocker())
        sys.path.insert(0, "src")
        os.environ["ECOM_DATA_SOURCE"] = "mock"

        import shopping_assistant.agent.tools
        import shopping_assistant.agent.shopping_agent
        import shopping_assistant.api.app
        import shopping_assistant.services.events
        import shopping_assistant.services.llm
        print("clean")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, cwd="."
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("clean")
