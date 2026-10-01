"""Turn-level supervised data with an exact loss mask.

One trajectory becomes one example per model decision, where the prompt is the
context the model actually saw at that turn. Training on whole trajectories
instead would teach the model to predict tool output, which it never generates.

The mask is verified against the chat template: if rendering the conversation
with the target appended does not equal prompt + target token by token, the
example is dropped rather than trained on a guess.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

from shoprl.harness.types import Message


class TemplateTokenizer(Protocol):
    """The slice of the HF tokenizer contract this module depends on."""

    def apply_chat_template(
        self,
        conversation: Sequence[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = ...,
        tokenize: bool = ...,
        add_generation_prompt: bool = ...,
    ) -> Any: ...


IGNORE_INDEX = -100


@dataclass(slots=True)
class TurnExample:
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    input_ids: list[int]
    labels: list[int]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "messages": self.messages,
            "tools": self.tools,
            "input_ids": self.input_ids,
            "labels": self.labels,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class SftSummary:
    trajectories: int = 0
    accepted: int = 0
    examples: int = 0
    rejected: int = 0
    reasons: dict[str, int] = field(default_factory=dict)
    token_mean: float = 0.0
    token_p95: int = 0
    token_max: int = 0
    target_mean: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "trajectories": self.trajectories,
            "accepted_trajectories": self.accepted,
            "turn_examples": self.examples,
            "rejected_examples": self.rejected,
            "rejection_reasons": dict(sorted(self.reasons.items())),
            "token_mean": round(self.token_mean, 1),
            "token_p95": self.token_p95,
            "token_max": self.token_max,
            "target_token_mean": round(self.target_mean, 1),
        }


def _as_conversation(messages: Sequence[Message]) -> list[dict[str, Any]]:
    conversation: list[dict[str, Any]] = []
    for message in messages:
        if message.role in {"system", "user"}:
            conversation.append({"role": message.role, "content": message.content})
        elif message.role == "assistant":
            conversation.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    **(
                        {
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "function": {
                                        "name": call.name,
                                        "arguments": call.arguments,
                                    },
                                }
                                for call in message.tool_calls
                            ]
                        }
                        if message.tool_calls
                        else {}
                    ),
                }
            )
        else:
            conversation.append(
                {"role": "tool", "content": message.content, "tool_call_id": message.tool_call_id}
            )
    return conversation


def _render(tokenizer: TemplateTokenizer, conversation, tools, **kwargs) -> list[int]:
    """Token ids for a conversation, whatever shape the chat template returns.

    transformers 5 returns a dict (``input_ids`` plus ``attention_mask``) where
    4.x returned a plain list. Iterating the dict yields its *keys*, which turns
    every target span into zero tokens and silently produces an empty dataset
    instead of an error, so the shape is normalised here.
    """
    rendered = tokenizer.apply_chat_template(conversation, tools=tools or None, **kwargs)
    if hasattr(rendered, "keys"):
        rendered = rendered["input_ids"]
    if hasattr(rendered, "tolist"):
        rendered = rendered.tolist()
    tokens = list(rendered)
    if tokens and isinstance(tokens[0], (list, tuple)):
        tokens = list(tokens[0])
    return [int(token) for token in tokens]


def build_turn_examples(
    record: dict[str, Any],
    tokenizer: TemplateTokenizer,
    *,
    tools: list[dict[str, Any]],
    max_tokens: int = 16384,
    verify_template: bool = True,
) -> tuple[list[TurnExample], list[dict[str, Any]]]:
    """Expand one trajectory record into one example per assistant turn."""
    examples: list[TurnExample] = []
    rejected: list[dict[str, Any]] = []
    trajectory_id = str(record.get("metadata", {}).get("session_id") or record.get("task_id"))
    targets = record.get("targets") or []
    contexts = record.get("contexts") or []

    for turn, (context, target) in enumerate(zip(contexts, targets)):
        example_id = f"{trajectory_id}:turn-{turn}"
        messages = list(context) + [target["message"]]
        prompt_ids = _render(tokenizer, context, tools, tokenize=True, add_generation_prompt=True)
        full_ids = _render(tokenizer, messages, tools, tokenize=True)
        target_ids = full_ids[len(prompt_ids) :]

        if verify_template and full_ids[: len(prompt_ids)] != list(prompt_ids):
            rejected.append({"example_id": example_id, "reason": "template_mismatch"})
            continue
        if not target_ids:
            rejected.append({"example_id": example_id, "reason": "empty_target"})
            continue
        if len(full_ids) > max_tokens:
            rejected.append({"example_id": example_id, "reason": "over_token_limit"})
            continue

        examples.append(
            TurnExample(
                messages=messages,
                tools=tools,
                input_ids=list(full_ids),
                labels=[IGNORE_INDEX] * len(prompt_ids) + list(target_ids),
                metadata={
                    "example_id": example_id,
                    "trajectory_id": trajectory_id,
                    "task_id": record.get("task_id"),
                    "turn": turn,
                    "reward": record.get("reward"),
                    "token_count": len(full_ids),
                    "target_token_count": len(target_ids),
                },
            )
        )
    return examples, rejected


def write_sft_dataset(
    records: list[dict[str, Any]],
    tokenizer: TemplateTokenizer,
    output_dir: Path,
    *,
    tools: list[dict[str, Any]],
    max_tokens: int = 16384,
    verify_template: bool = True,
) -> SftSummary:
    """Write ``turn_examples.jsonl`` plus a summary for one SFT run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = SftSummary(trajectories=len(records))
    token_counts: list[int] = []
    target_counts: list[int] = []

    with (
        (output_dir / "turn_examples.jsonl").open("w", encoding="utf-8") as examples_file,
        (output_dir / "rejected_turn_examples.jsonl").open("w", encoding="utf-8") as rejected_file,
    ):
        for record in records:
            if not record.get("metadata", {}).get("accepted", True):
                summary.reasons["trajectory_rejected"] = summary.reasons.get("trajectory_rejected", 0) + 1
                continue
            summary.accepted += 1
            examples, rejected = build_turn_examples(
                record,
                tokenizer,
                tools=tools,
                max_tokens=max_tokens,
                verify_template=verify_template,
            )
            for example in examples:
                examples_file.write(json.dumps(example.to_json(), ensure_ascii=False) + "\n")
                token_counts.append(example.metadata["token_count"])
                target_counts.append(example.metadata["target_token_count"])
            for item in rejected:
                rejected_file.write(json.dumps(item, ensure_ascii=False) + "\n")
                summary.reasons[item["reason"]] = summary.reasons.get(item["reason"], 0) + 1
            summary.examples += len(examples)
            summary.rejected += len(rejected)

    token_counts.sort()
    summary.token_mean = sum(token_counts) / len(token_counts) if token_counts else 0.0
    summary.token_p95 = token_counts[min(len(token_counts) - 1, int(0.95 * len(token_counts)))] if token_counts else 0
    summary.token_max = token_counts[-1] if token_counts else 0
    summary.target_mean = sum(target_counts) / len(target_counts) if target_counts else 0.0
    (output_dir / "summary.json").write_text(
        json.dumps(summary.to_json(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary
