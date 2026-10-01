"""Tool-call parsing: the failure mode that silently ends every rollout."""

from __future__ import annotations

from shoprl.train.parsing import parse_tool_calls, strip_tool_calls


def test_a_single_call_in_the_standard_wrapper_is_parsed():
    calls = parse_tool_calls('<tool_call>\n{"name": "shop_act", "arguments": {"action": "click[x]"}}\n</tool_call>')

    assert len(calls) == 1
    assert calls[0].name == "shop_act"
    assert calls[0].arguments == {"action": "click[x]"}
    assert calls[0].id == "call_0"


def test_arguments_sent_as_a_json_string_are_decoded():
    calls = parse_tool_calls('<tool_call>{"name": "shop_act", "arguments": "{\\"action\\": \\"search[cup]\\"}"}</tool_call>')
    assert calls[0].arguments == {"action": "search[cup]"}


def test_several_calls_in_one_turn_are_all_returned():
    text = (
        '<tool_call>{"name": "shop_reset", "arguments": {}}</tool_call>\n'
        '<tool_call>{"name": "shop_act", "arguments": {"action": "search[a]"}}</tool_call>'
    )
    assert [call.name for call in parse_tool_calls(text)] == ["shop_reset", "shop_act"]


def test_a_call_inside_a_fenced_block_is_still_found():
    text = '<tool_call>```json\n{"name": "shop_act", "arguments": {"action": "click[buy now]"}}\n```</tool_call>'
    assert parse_tool_calls(text)[0].arguments["action"] == "click[buy now]"


def test_prose_without_a_call_produces_no_calls():
    """Ending the episode is the correct outcome for text with no action in it."""
    assert parse_tool_calls("I think we should buy the kettle.") == []
    assert parse_tool_calls("") == []


def test_a_malformed_block_is_skipped_rather_than_guessed_at():
    assert parse_tool_calls('<tool_call>{"name": "shop_act", "arguments": {</tool_call>') == []


def test_stripping_leaves_the_prose_and_drops_the_call():
    text = '<tool_call>{"name": "shop_act", "arguments": {}}</tool_call>\n\n推荐这一款。'
    assert strip_tool_calls(text) == "推荐这一款。"

