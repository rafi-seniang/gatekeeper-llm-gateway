"""Masks structured identifiers before a request leaves the building.

These are formats, not language: an email address, a phone number, a 16-digit NIK or a card number has a fixed
shape, so a pattern finds it reliably. Meaning (is this message about a person?) is left to the models.
"""

import re

IDENTIFIER_PATTERNS = {
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"),
    # a card as printed: four groups of four digits
    "CARD": re.compile(r"\b\d{4}(?:[ -]\d{4}){3}\b"),
    # an Indonesian NIK: sixteen digits in a row (a card written without spaces is masked here too)
    "NIK": re.compile(r"\b\d{16}\b"),
    # a phone number starts with + and a country code, or with 0, so dates and amounts are left alone
    "PHONE": re.compile(r"(?<![\w+])(?:\+\d{1,3}[ -]?|0)\d{2,4}(?:[ -]?\d{3,4}){2,3}\b"),
}


def mask_identifiers(text: str) -> tuple[str, int]:
    """The text with each identifier replaced by its kind in brackets, and how many were replaced."""
    replaced = 0
    for kind, pattern in IDENTIFIER_PATTERNS.items():  # four patterns, each depends on the text the last left
        text, count = pattern.subn(f"[{kind}]", text)
        replaced += count
    return text, replaced
