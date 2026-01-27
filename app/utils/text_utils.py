# app/utils/text_utils.py
from __future__ import annotations

import re
import string

from app.nlu.intent_patterns.add_item import ITEM_SEPARATOR_PAT

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")
_LEADING_TRAILING_RE = re.compile(r"^\s+|\s+$")


def normalize_text(text: str) -> str:
    """
    Normalizes raw STT text into a deterministic form.

    Steps:
    - Lowercase
    - Remove punctuation
    - Collapse whitespace
    - Trim
    """

    if not text:
        return ""

    text = text.lower().strip()
    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # normalize whitespace
    text = re.sub(r"\s+", " ", text)
    text = _PUNCT_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    text = _LEADING_TRAILING_RE.sub("", text)

    return text


def split_candidates(text: str) -> list[str]:
    """
    Split user input into candidate chunks using lexical separators.
    Deterministic. No semantics.
    """
    if not text:
        return []

    parts = ITEM_SEPARATOR_PAT.split(text)
    return [p.strip() for p in parts if p.strip()]