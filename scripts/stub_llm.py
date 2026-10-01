#!/usr/bin/env python3
"""A canned OpenAI-compatible endpoint, for running the app without a provider.

Useful for two things: developing the UI without spending API credit, and
proving the server works end to end in CI. The app's real client, tool registry
and SSE framing all run; only the model is canned. The second turn picks a
product id out of the search result it was just handed, so the flow matches what
a real model produces.

    python scripts/stub_llm.py --port 9099
    ECOM_LLM_BASE_URL=http://127.0.0.1:9099/v1 ECOM_LLM_MODEL=stub \
        ECOM_LLM_API_KEY=stub python -m shopping_assistant
"""

from __future__ import annotations

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, HTTPServer

# A catalogue row rendered by the search tool starts with "id | name | ...".
PRODUCT_ID = re.compile(r"^(\S+) \|", re.MULTILINE)
REPLY_ZH = "我推荐这一款，符合你的预算。"
REPLY_EN = "This one fits what you asked for."


def tool_call(name: str, **arguments) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": f"call_{name}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }


def plan(messages: list[dict], reply: str) -> dict:
    """Search, then present what the search returned, then answer."""
    calls = [m for m in messages if m.get("role") == "assistant" and m.get("tool_calls")]
    if not calls:
        return tool_call("search_catalog", query="香薰", max_price=300, limit=3)
    if len(calls) == 1:
        results = [m for m in messages if m.get("role") == "tool"]
        ids = PRODUCT_ID.findall(str(results[-1].get("content") or "")) if results else []
        if ids:
            return tool_call("present_recommendation", product_ids=ids[:2])
    return {"role": "assistant", "content": reply}


def handler_for(reply: str):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            payload = json.dumps(
                {
                    "id": "stub",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": plan(body.get("messages") or [], reply),
                            "finish_reason": "stop",
                        }
                    ],
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args) -> None:
            pass

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=9099)
    parser.add_argument("--language", choices=["zh", "en"], default="zh")
    args = parser.parse_args()

    reply = REPLY_ZH if args.language == "zh" else REPLY_EN
    print(f"stub llm on http://127.0.0.1:{args.port}/v1", flush=True)
    HTTPServer(("127.0.0.1", args.port), handler_for(reply)).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
