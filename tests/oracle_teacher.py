"""A scripted teacher for tests that need a rollout without a model.

It plays the page it was just shown, like a real agent, and reports the token
ids and log-probabilities a real engine would report for each reply. That makes
it usable wherever a rollout has to look exactly like a live one — the RL sample
conversion, the GRPO loop — while costing no GPU and no API call.
"""

from __future__ import annotations

import json

from shoprl.harness.types import Message, ModelResponse, ToolCall

NAVIGATION = {"buy now", "< prev", "back to search", "description", "features", "reviews"}
MARKER = "可点击的按钮: "


def clickables_of(page: str) -> list[str]:
    if MARKER not in page:
        return []
    try:
        return json.loads(page.split(MARKER, 1)[1].strip())
    except json.JSONDecodeError:
        return []


class OracleTeacher:
    """Plays one task correctly, emitting replies shaped like a live engine's."""

    def __init__(self, catalogue, tokenizer, task_id: int, *, solve: bool = True):
        self.tokenizer = tokenizer
        self.target = catalogue.at(task_id)
        self.solve = solve
        self.phase = "search"
        self.satisfied: set[int] = set()

    def complete(self, messages: list[Message], tools=None, **_kwargs) -> ModelResponse:
        from shoprl.env.reward import _ratio

        # A rollout always opens with shop_reset and no observations yet. Keying
        # on that rather than on construction means one teacher can serve many
        # rollouts in a row, which is what a group of candidates needs.
        if not any(message.role == "tool" for message in messages):
            self.phase = "search"
            self.satisfied = set()
            return self._response(messages, "shop_reset")

        page = next((m.content for m in reversed(messages) if m.role == "tool"), "")
        clickables = clickables_of(page)
        asin = self.target.asin.lower()

        if self.phase in {"search", "paging"}:
            if asin in clickables:
                self.phase = "item"
                return self._response(messages, "shop_act", action=f"click[{asin}]")
            if self.phase == "search":
                self.phase = "paging"
                return self._response(
                    messages, "shop_act", action=f"search[{self.target.instruction}]"
                )
            if "next >" in clickables:
                return self._response(messages, "shop_act", action="click[next >]")
            return ModelResponse(text="the target is not in these results")

        if self.phase == "item":
            options = [value for value in clickables if value not in NAVIGATION]
            for index, wanted in enumerate(self.target.instruction_options):
                if index in self.satisfied or not options:
                    continue
                score, best = max(
                    ((_ratio(value, wanted.lower()), value) for value in options),
                    default=(0.0, ""),
                )
                if score > 85:
                    self.satisfied.add(index)
                    return self._response(messages, "shop_act", action=f"click[{best}]")
            if "buy now" in clickables:
                self.phase = "done"
                return self._response(messages, "shop_act", action="click[buy now]")

        return ModelResponse(text="no move left")

    def _response(self, messages: list[Message], name: str, **arguments) -> ModelResponse:
        """Sample the continuation the way an engine does: full rendering minus prompt."""
        from shoprl.data.sft import _render
        from shoprl.train.grpo import TOOLS

        conversation = [message.as_dict() for message in messages]
        assistant = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"type": "function", "function": {"name": name, "arguments": arguments}}
            ],
        }
        prompt_ids = _render(
            self.tokenizer, conversation, TOOLS, tokenize=True, add_generation_prompt=True
        )
        full_ids = _render(self.tokenizer, conversation + [assistant], TOOLS, tokenize=True)
        ids = list(full_ids[len(prompt_ids) :])
        return ModelResponse(
            tool_calls=[ToolCall(id=name, name=name, arguments=arguments)],
            token_ids=ids,
            # The value is irrelevant; only that it is present and aligned.
            logprobs=[-0.1] * len(ids),
        )


class ReluctantTeacher(OracleTeacher):
    """Solves every other rollout, so a candidate group has reward variance.

    Group-relative advantages are zero when every candidate scores the same, so a
    test that wants the optimiser touched has to produce a spread. Alternating on
    the rollout index is enough and needs no model.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rollouts = 0
        self.give_up = False

    def complete(self, messages: list[Message], tools=None, **kwargs) -> ModelResponse:
        if not any(message.role == "tool" for message in messages):
            self.rollouts += 1
            self.give_up = self.rollouts % 2 == 0
        if self.give_up:
            return ModelResponse(text="I would rather not.")
        return super().complete(messages, tools, **kwargs)
