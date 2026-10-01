"""Turn-level SFT conversion: loss mask, template verification, rejection reasons."""

from __future__ import annotations

from shoprl.data.sft import IGNORE_INDEX, build_turn_examples, write_sft_dataset

HISTORY = ["-1", "-1"]


class FakeTokenizer:
    """Renders a conversation as a flat id list; a conversation stays a prefix."""

    def __init__(self, *, break_prefix: bool = False):
        self.break_prefix = break_prefix

    def apply_chat_template(self, conversation, *, tools=None, tokenize=True, add_generation_prompt=False):
        text = "|".join(f"{message['role']}:{message.get('content', '')}" for message in conversation)
        if add_generation_prompt:
            text += "|assistant:"
        if self.break_prefix:
            text += "|extra"
        return [ord(char) % 97 for char in text]


def record(contexts, targets) -> dict:
    return {
        "task_id": 3,
        "reward": 0.5,
        "metadata": {"session_id": "abc", "accepted": True},
        "contexts": contexts,
        "targets": targets,
    }


def target(content: str) -> dict:
    return {"message": {"role": "assistant", "content": content}}


def test_prompt_tokens_are_masked_and_target_tokens_are_trained():
    data = record(
        [
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}],
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "task"},
                {"role": "assistant", "content": "search[cup]"},
            ],
        ],
        [target("search[cup]"), target("click[buy now]")],
    )

    examples, rejected = build_turn_examples(data, FakeTokenizer(), tools=[])

    assert rejected == [] and len(examples) == 2
    for example in examples:
        assert len(example.input_ids) == len(example.labels)
        trained = [label for label in example.labels if label != IGNORE_INDEX]
        assert trained
        # Every trained position is in the target span, and the prompt is fully masked.
        first_trained = next(i for i, label in enumerate(example.labels) if label != IGNORE_INDEX)
        assert example.labels[:first_trained] == [IGNORE_INDEX] * first_trained
        assert example.labels[first_trained:] == example.input_ids[first_trained:]


def test_a_template_that_is_not_prefix_consistent_is_rejected():
    data = record(
        [[{"role": "user", "content": "task"}]],
        [target("search[cup]")],
    )

    examples, rejected = build_turn_examples(
        data, FakeTokenizer(break_prefix=True), tools=[], verify_template=True
    )

    assert examples == []
    assert rejected == [{"example_id": "abc:turn-0", "reason": "template_mismatch"}]


def test_over_long_examples_are_dropped_rather_than_truncated():
    data = record(
        [[{"role": "user", "content": "task"}]],
        [target("x" * 200)],
    )
    examples, rejected = build_turn_examples(data, FakeTokenizer(), tools=[], max_tokens=50)
    assert examples == []
    assert rejected[0]["reason"] == "over_token_limit"


def test_write_sft_dataset_summarises_what_it_kept(tmp_path):
    data = record(
        [[{"role": "user", "content": "task"}]],
        [target("search[cup]")],
    )
    rejected = {**data, "metadata": {"session_id": "def", "accepted": False}}

    summary = write_sft_dataset([data, rejected], FakeTokenizer(), tmp_path, tools=[])

    assert summary.trajectories == 2
    assert summary.accepted == 1
    assert summary.examples == 1
    assert summary.reasons == {"trajectory_rejected": 1}
    assert (tmp_path / "turn_examples.jsonl").exists()
    assert (tmp_path / "summary.json").exists()

