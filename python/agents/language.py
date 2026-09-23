"""Language helpers for prompt directives."""

from __future__ import annotations

from collections.abc import Sequence

LANGUAGE_NAMES = {"en": "English", "zh": "Simplified Chinese"}


def _is_mostly_cjk(values: Sequence[str]) -> bool:
    return bool(values) and sum(
        1 for value in values if any("\u4e00" <= char <= "\u9fff" for char in value)
    ) * 2 >= len(values)


def catalog_language_rule(categories: Sequence[str]) -> str:
    """Tell the planner which language the loaded catalog speaks.

    The downloaded ShopSimulator catalog is Chinese while the built-in demo
    catalog is English, so search terms have to follow whichever is active even
    when the shopper writes in the other language.
    """
    language = "Chinese" if _is_mostly_cjk(categories) else "English"
    return (
        f"search.keywords and search.category MUST be {language} catalog terms "
        "(translate the shopper's wording when necessary); only `reply` follows the reply language."
    )


def language_directive(code: str) -> str:
    """Instruction appended to system prompts so replies follow the UI language."""
    return f"Always write your reply in {LANGUAGE_NAMES.get(code, 'English')}."
