"""One shopper's interaction with the store, mirroring upstream's page flow.

Pages: search -> results (20 per page) -> item (options selectable) -> optional
description/features/reviews -> purchase. ``search[..]`` and ``click[..]`` are
the only actions; an action that is not on the current page changes nothing,
exactly as upstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from shoprl.env.actions import (
    BACK_TO_SEARCH,
    END_BUTTON,
    MAX_HISTORY_LENGTH,
    NEXT_PAGE,
    PREV_PAGE,
    PRODUCT_WINDOW,
    SEARCH_BUTTON,
    SEARCH_RETURN_N,
    SUB_PAGE_KEYS,
    SUB_PAGES,
    parse_action,
    render,
    render_available_actions,
)
from shoprl.env.catalog import Catalog, Product, ProductOption
from shoprl.env.reward import (
    Goal,
    Purchase,
    deterministic_price,
    deterministic_price_upper,
    score_purchase,
)
from shoprl.env.search import Bm25Index


@dataclass(slots=True)
class Outcome:
    """Everything one environment step reports back."""

    text: str
    done: bool = False
    reward: float = 0.0
    reward_detail: dict[str, float] = field(default_factory=dict)
    purchase: dict[str, Any] | None = None
    goal: dict[str, Any] | None = None
    over: bool = False
    changed: bool = True


class ShopSession:
    """Mutable per-rollout state; one instance serves exactly one task."""

    def __init__(
        self,
        catalog: Catalog,
        index: Bm25Index,
        product: Product,
        *,
        history_limit: int = MAX_HISTORY_LENGTH,
    ):
        self.catalog = catalog
        self.index = index
        self.product = product
        self.history_limit = history_limit

        self.price = deterministic_price(product.asin, product.pricing)
        self.goal = Goal(
            asin=product.asin,
            title=product.title,
            category=product.category,
            instruction=product.instruction,
            attributes=list(product.instruction_attributes or product.attributes),
            options=list(product.instruction_options),
            price_upper=deterministic_price_upper(product.asin, product.instruction, self.price),
            query=product.query,
        )

        self.page = 1
        self.keywords: list[str] = []
        self.results: list[int] = []
        self.asin = ""
        self.options: dict[str, str] = {}
        self.view = "start"
        self.history: list[str] = []
        self.reward = 0.0
        self.reward_detail: dict[str, float] = {}
        self.purchase: dict[str, Any] | None = None
        self.done = False

    # ── public API ────────────────────────────────────────────────────

    def reset(self) -> Outcome:
        self.page, self.keywords, self.asin, self.results = 1, [], "", []
        self.options, self.view, self.history = {}, "start", []
        self.reward, self.reward_detail, self.purchase, self.done = 0.0, {}, None, False
        return self._observe()

    def step(self, text: str) -> Outcome:
        action = parse_action(text)
        self.history.append(text)
        changed = False

        if action.name == "search" and action.argument and self.view != "done":
            changed = self._search(action.argument)
        elif action.name == "click" and action.argument is not None:
            changed = self._click(action.argument.lower())

        outcome = self._observe()
        outcome.changed = changed
        outcome.over = outcome.over or len(self.history) > self.history_limit
        return outcome

    # ── actions ───────────────────────────────────────────────────────

    def _search(self, keywords: str) -> bool:
        self.keywords = [term for term in keywords.split(" ") if term]
        self.results = self.index.search(self.keywords, limit=SEARCH_RETURN_N)
        self.page, self.asin, self.options, self.view = 1, "", {}, "results"
        return True

    def _click(self, value: str) -> bool:
        if value not in self.clickables():
            return False

        if self.view == "results":
            return self._click_results(value)
        if self.view == "item":
            return self._click_item(value)
        if self.view in SUB_PAGE_KEYS:
            if value == PREV_PAGE.lower():
                self.view = "item"
                return True
            if value == BACK_TO_SEARCH.lower():
                self.view = "start"
                return True
        return False

    def _click_results(self, value: str) -> bool:
        if value == BACK_TO_SEARCH.lower():
            self.view = "start"
            return True
        if value == NEXT_PAGE.lower():
            self.page += 1
            return True
        if value == PREV_PAGE.lower() and self.page > 1:
            self.page -= 1
            return True
        product = self.catalog.get(value.upper())
        if product is None:
            return False
        self.asin, self.options, self.view = product.asin, {}, "item"
        return True

    def _click_item(self, value: str) -> bool:
        if value == BACK_TO_SEARCH.lower():
            self.view = "start"
            return True
        if value == PREV_PAGE.lower():
            self.view = "results"
            return True
        if value in SUB_PAGE_KEYS:
            self.view = value
            return True
        if value == END_BUTTON.lower():
            self._purchase()
            return True
        for name, entries in self._product_options().items():
            for option in entries:
                if option.value.lower() == value:
                    self.options[name] = option.value
                    return True
        return False

    def _purchase(self) -> None:
        product = self.catalog.get(self.asin)
        if product is None:
            return
        purchase = Purchase(
            asin=product.asin,
            title=product.title,
            category=product.category,
            price=self.price,
            attributes=list(product.attributes),
            options=list(self.options.values()),
            query=product.query,
        )
        self.reward, self.reward_detail = score_purchase(purchase, self.goal)
        self.purchase = {
            "asin": product.asin,
            "title": product.title,
            "options": dict(self.options),
        }
        self.done, self.view = True, "done"

    # ── rendering ─────────────────────────────────────────────────────

    def clickables(self) -> list[str]:
        if self.view == "start":
            return [SEARCH_BUTTON.lower()]
        if self.view == "results":
            values = [BACK_TO_SEARCH.lower()]
            if self.page > 1:
                values.append(PREV_PAGE.lower())
            values.append(NEXT_PAGE.lower())
            return values + [asin.lower() for asin in self._page_asins()]
        if self.view == "item":
            values = [BACK_TO_SEARCH.lower(), PREV_PAGE.lower(), *SUB_PAGE_KEYS]
            values += [
                option.value.lower()
                for entries in self._product_options().values()
                for option in entries
            ]
            return values + [END_BUTTON.lower()]
        if self.view in SUB_PAGE_KEYS:
            return [PREV_PAGE.lower(), BACK_TO_SEARCH.lower()]
        return []

    def has_search_bar(self) -> bool:
        return self.view != "done"

    def observation(self) -> str:
        renderers = {
            "start": self._start_lines,
            "results": self._results_lines,
            "item": self._item_lines,
            "done": self._done_lines,
        }
        return render(renderers.get(self.view, self._sub_page_lines)())

    def _observe(self) -> Outcome:
        text = self.observation() + render_available_actions(self.has_search_bar(), self.clickables())
        return Outcome(
            text=text,
            done=self.done,
            reward=self.reward,
            reward_detail=dict(self.reward_detail),
            purchase=self.purchase,
            goal=self.goal_payload() if self.done else None,
        )

    def _start_lines(self) -> list[str]:
        return ["WebShop", f"Instruction: {self.goal.instruction}", SEARCH_BUTTON]

    def _results_lines(self) -> list[str]:
        lines = [
            f"Instruction: {self.goal.instruction}",
            BACK_TO_SEARCH,
            f"Page {self.page} (Total results: {len(self.results)})",
        ]
        lines += [PREV_PAGE] if self.page > 1 else []
        lines.append(NEXT_PAGE)
        for product in self._page_products():
            lines += [product.asin, product.title, product.price_tag]
        return lines

    def _item_lines(self) -> list[str]:
        product = self.catalog.get(self.asin)
        if product is None:
            return self._results_lines()
        lines = [f"Instruction: {self.goal.instruction}", BACK_TO_SEARCH, PREV_PAGE]
        for name, entries in self._product_options().items():
            lines.append(name)
            lines += [option.value for option in entries]
        lines += [
            product.title,
            f"价格: {self._display_price(product)}",
            f"店铺: {product.shop_name}",
            *SUB_PAGES,
            END_BUTTON,
        ]
        return lines

    def _sub_page_lines(self) -> list[str]:
        product = self.catalog.get(self.asin)
        if product is None:
            return []
        body = {
            "description": product.description,
            "features": " ".join(product.attributes),
            "reviews": "",
        }.get(self.view, "")
        return [f"Instruction: {self.goal.instruction}", PREV_PAGE, BACK_TO_SEARCH, body]

    def _done_lines(self) -> list[str]:
        return [
            "Thank you for shopping with us!",
            f"Your score (min 0.0, max 1.0) {self.reward}",
        ]

    # ── helpers ───────────────────────────────────────────────────────

    def _product_options(self) -> dict[str, list[ProductOption]]:
        product = self.catalog.get(self.asin)
        return product.options if product else {}

    def _display_price(self, product: Product) -> float:
        if not self.options:
            return self.price
        option_price = product.option_price(next(reversed(self.options.values())))
        return option_price if option_price is not None else self.price

    def _page_asins(self) -> list[str]:
        start = (self.page - 1) * PRODUCT_WINDOW
        window = self.results[start : start + PRODUCT_WINDOW]
        return [product.asin for product in (self._position(index) for index in window) if product]

    def _page_products(self) -> list[Product]:
        return [product for product in (self._position(index) for index in self._page_window()) if product]

    def _page_window(self) -> list[int]:
        start = (self.page - 1) * PRODUCT_WINDOW
        return self.results[start : start + PRODUCT_WINDOW]

    def _position(self, index: int) -> Product | None:
        return self.catalog.products[index] if 0 <= index < len(self.catalog.products) else None

    def goal_payload(self) -> dict[str, Any]:
        return {
            "asin": self.goal.asin,
            "title": self.goal.title,
            "category": self.goal.category,
            "attributes": list(self.goal.attributes),
            "options": list(self.goal.options),
            "price_upper": self.goal.price_upper,
            "instruction": self.goal.instruction,
        }

