"""Language helpers for prompt directives."""

from __future__ import annotations

LANGUAGE_NAMES = {"en": "English", "zh": "Simplified Chinese"}

# The demo catalog is stored in English, so search terms must stay in English
# even when the shopper writes in another language.
CATALOG_LANGUAGE_RULE = (
    "search.keywords and search.category MUST be English catalog terms "
    "(translate the shopper's wording when necessary); only `reply` follows the reply language."
)


def language_directive(code: str) -> str:
    """Instruction appended to system prompts so replies follow the UI language."""
    return f"Always write your reply in {LANGUAGE_NAMES.get(code, 'English')}."
