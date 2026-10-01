"""The ShopSimulator action grammar and page-text rendering.

Upstream renders an HTML storefront and flattens it to text for the agent: every
visible string is joined with `` [SEP] ``. This module produces that text
directly, which keeps the observation a pure function of session state and drops
the templating layer without changing what the model reads.

Deliberate divergence: upstream's HTML-to-text pass also emits whitespace-only
nodes, so its observations contain runs of empty `` [SEP] `` separators. Those
carry no signal and are dropped here.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

SEP = " [SEP] "

END_BUTTON = "Buy Now"
NEXT_PAGE = "Next >"
PREV_PAGE = "< Prev"
BACK_TO_SEARCH = "Back to Search"
SUB_PAGES = ("Description", "Features", "Reviews")
SUB_PAGE_KEYS = tuple(page.lower() for page in SUB_PAGES)
SEARCH_BUTTON = "Search"

# Upstream constants from shop_env/web_agent_site/engine/engine.py
SEARCH_RETURN_N = 150
PRODUCT_WINDOW = 20
MAX_HISTORY_LENGTH = 42

_ACTION = re.compile(r"(.+)\[(.+)\]", re.DOTALL)


@dataclass(slots=True)
class Action:
    name: str
    argument: str | None


def parse_action(text: str) -> Action:
    """Parse ``search[kw]`` / ``click[value]``; anything else is inert."""
    match = _ACTION.match((text or "").strip())
    if match is None:
        return Action(name=(text or "").strip().lower(), argument=None)
    return Action(name=match.group(1).strip().lower(), argument=match.group(2).strip())


def render(lines: list[str]) -> str:
    return SEP.join(line for line in lines if line)


def render_available_actions(has_search_bar: bool, clickables: list[str]) -> str:
    """The trailing block upstream appends to every observation."""
    return (
        f"\n\n搜索功能是否可用: {has_search_bar}"
        f"\n\n可点击的按钮: {json.dumps(clickables, ensure_ascii=False)}"
    )
