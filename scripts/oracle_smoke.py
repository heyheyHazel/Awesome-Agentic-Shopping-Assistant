"""Self-check: drive the whole stack with an oracle and report the ceiling.

An oracle that knows the target product is the cheapest end-to-end test there
is. It exercises the loop, the tools, the environment and the metric code on real
data with no model, no API key and no GPU, and it answers a question no unit test
can: is the task pool solvable at all, and where does it stop being solvable?

    python scripts/oracle_smoke.py --limit 120

Anything much below the numbers in docs/data-audit.md means the environment or
the task pools changed, not that the oracle got unlucky.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from shoprl.env.catalog import load_catalog
from shoprl.env.local import EnvPool
from shoprl.env.reward import _ratio
from shoprl.env.search import Bm25Index
from shoprl.eval.metrics import summarise
from shoprl.harness.rollout import EpisodeRunner
from shoprl.harness.types import ModelResponse, ToolCall

NAVIGATION = {
    "buy now",
    "< prev",
    "back to search",
    "description",
    "features",
    "reviews",
    "next >",
}


def clickables_of(page: str) -> list[str]:
    marker = "可点击的按钮: "
    if marker not in page:
        return []
    try:
        return json.loads(page.split(marker, 1)[1].strip())
    except json.JSONDecodeError:
        return []


class Oracle:
    """Plays the right move by reading the page it was just shown."""

    def __init__(self, catalog, task_id: int):
        self.target = catalog.at(task_id)
        self.phase = "reset"
        self.satisfied: set[int] = set()

    def complete(self, messages, tools=None, **kwargs) -> ModelResponse:
        page = next((m.content for m in reversed(messages) if m.role == "tool"), "")
        clickables = clickables_of(page)
        asin = self.target.asin.lower()

        if self.phase == "reset":
            self.phase = "search"
            return self._call("shop_reset")

        if self.phase in {"search", "paging"}:
            if asin in clickables:
                self.phase = "item"
                return self._call("shop_act", action=f"click[{asin}]")
            if self.phase == "search":
                self.phase = "paging"
                return self._call("shop_act", action=f"search[{self.target.instruction}]")
            if "next >" in clickables:
                return self._call("shop_act", action="click[next >]")
            return ModelResponse(text="the target is not in these results")

        if self.phase == "item":
            options = [value for value in clickables if value not in NAVIGATION]
            for index, wanted in enumerate(self.target.instruction_options):
                if index in self.satisfied:
                    continue
                score, best = max(
                    ((_ratio(value, wanted.lower()), value) for value in options),
                    default=(0.0, ""),
                )
                if score > 85:
                    self.satisfied.add(index)
                    return self._call("shop_act", action=f"click[{best}]")
            if "buy now" in clickables:
                self.phase = "done"
                return self._call("shop_act", action="click[buy now]")

        return ModelResponse(text="the oracle has no move left")

    @staticmethod
    def _call(name: str, **arguments) -> ModelResponse:
        return ModelResponse(tool_calls=[ToolCall(id=name, name=name, arguments=arguments)])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=120, help="how many consecutive task ids to play")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--capacity", type=int, default=4)
    parser.add_argument("--json", action="store_true", help="print the raw metrics object")
    args = parser.parse_args()

    catalog = load_catalog()
    pool = EnvPool(catalog, Bm25Index.build(catalog), capacity=args.capacity)
    records = []
    for task_id in range(args.offset, args.offset + args.limit):
        runner = EpisodeRunner(Oracle(catalog, task_id), pool, max_turns=20, max_tool_calls=40)
        records.append(runner.run(task_id).to_json())

    metrics = summarise(records)
    if args.json:
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return 0

    print(f"tasks                 {metrics['samples']}")
    print(f"finished an episode   {metrics['done_rate']:.1%}")
    print(f"bought the right item {metrics['right_product_rate']:.1%}")
    print(f"r_loose               {metrics['r_loose']:.3f}")
    print(f"r_hard                {metrics['r_hard']:.3f}")
    print(f"r_success             {metrics['r_success']:.3f}")
    print(f"terminations          {metrics['terminations']}")
    print(f"pool                  {pool.status()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

