"""An offline teacher: finishes a shopping task without a model.

Used in two places. In tests it stands in for a policy, so the RL sample
conversion and the GRPO loop can be exercised without a GPU. From the command
line it backs ``collect --teacher scripted``, which walks the entire collection
path — task pool, environment, harness, trajectory writing — before anyone
spends API credit on the real thing.

It is deliberately not a good shopper: it plays the page it was shown. That is
enough to finish a task and produce a well-formed trajectory, which is all a dry
run is for.
"""

from __future__ import annotations

import json
from typing import Any

from shoprl.harness.types import Message, ModelResponse, ToolCall

NAVIGATION = {"buy now", "< prev", "back to search", "description", "features", "reviews"}
CLICKABLE_MARKER = "可点击的按钮: "


def clickables_of(page: str) -> list[str]:
    """The clickable values the environment appends to every observation."""
    if CLICKABLE_MARKER not in page:
        return []
    try:
        return json.loads(page.split(CLICKABLE_MARKER, 1)[1].strip())
    except json.JSONDecodeError:
        return []


class ScriptedTeacher:
    """Plays one task correctly, emitting replies shaped like a live engine's."""

    def __init__(self, catalogue: Any, task_id: int, tokenizer: Any = None):
        self.catalogue = catalogue
        self.tokenizer = tokenizer
        self.target = catalogue.at(task_id)
        self.phase = "search"
        self.satisfied: set[int] = set()

    def complete(self, messages: list[Message], tools=None, **_kwargs) -> ModelResponse:
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
            from shoprl.env.reward import _ratio

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
        """A tool call, with the token ids an engine would have sampled if asked.

        Collection does not need the tokens — SFT re-renders the text later — so
        a dry run works without a tokenizer, and therefore without a checkpoint.
        """
        call = ToolCall(id=name, name=name, arguments=arguments)
        if self.tokenizer is None:
            return ModelResponse(tool_calls=[call])

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
            tool_calls=[call],
            token_ids=ids,
            # The value is irrelevant to callers; only that it is present and
            # aligned, which is what the RL sample check verifies.
            logprobs=[-0.1] * len(ids),
        )


class ReluctantTeacher(ScriptedTeacher):
    """Solves every other rollout, so a candidate group has reward variance.

    Group-relative advantages are zero when every candidate scores the same, so a
    caller that wants the optimiser touched has to produce a spread. Alternating
    on the rollout index is enough and needs no model.
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


def scripted_runner(catalogue: Any, pool: Any, **kwargs):
    """An ``EpisodeRunner`` whose teacher is rebuilt for each task.

    A teacher is bound to one task, so a run over a pool needs a new one per
    rollout rather than a single instance reused.
    """
    from shoprl.harness.rollout import EpisodeRunner

    class Runner(EpisodeRunner):
        def run(self, task_id: int, **run_kwargs):
            self.backend = ScriptedTeacher(catalogue, task_id)
            return super().run(task_id, **run_kwargs)

    return Runner(ScriptedTeacher(catalogue, 0), pool, **kwargs)
