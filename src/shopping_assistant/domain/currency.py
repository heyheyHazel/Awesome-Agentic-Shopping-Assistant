"""Currency helpers: ISO code to display symbol."""

from __future__ import annotations

from shopping_assistant.settings import get_settings

SYMBOLS = {
    "USD": "$",
    "CNY": "¥",
    "EUR": "€",
    "GBP": "£",
    "JPY": "¥",
    "HKD": "HK$",
}


def currency_code() -> str:
    """Active currency code from settings (default CNY to match the catalog)."""
    return get_settings().currency.strip().upper() or "CNY"


def currency_symbol() -> str:
    """Display symbol for the active currency, falling back to the code."""
    code = currency_code()
    return SYMBOLS.get(code, f"{code} ")
