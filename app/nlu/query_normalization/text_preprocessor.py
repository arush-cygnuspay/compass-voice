# app/nlu/query_normalization/text_preprocessor.py
from __future__ import annotations

import re
import string
import unicodedata
from dataclasses import dataclass
from typing import Optional

_TRANSLATION_TABLE = str.maketrans("", "", string.punctuation)

_RE_WS = re.compile(r"\s+")
_RE_DOTS = re.compile(r"\.{2,}")
_RE_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?])")
_RE_MULTI_PUNCT = re.compile(r"([!?.,;:])\1{1,}")
_RE_ELONGATION = re.compile(r"(.)\1{3,}", re.IGNORECASE)
_RE_ASR_BRACKETS = re.compile(
    r"[\[\(]\s*(noise|music|silence|inaudible|crosstalk|laughter)\s*[\]\)]",
    re.IGNORECASE,
)

_INVISIBLE_CHARS = {
    "\u200b",
    "\u200c",
    "\u200d",
    "\ufeff",
}

_FILLER_PATTERNS = [
    r"\bum+\b",
    r"\buh+\b",
    r"\ber+\b",
    r"\bah+\b",
    r"\bhmm+\b",
    r"\byou know\b",
    r"\bi mean\b",
]
_RE_FILLERS = re.compile("|".join(_FILLER_PATTERNS), re.IGNORECASE)

_PUNCT_WORDS = {
    "comma": ",",
    "period": ".",
    "fullstop": ".",
    "full stop": ".",
    "question mark": "?",
    "exclamation point": "!",
    "exclamation mark": "!",
}

_WAITING_PREFIX_PATTERNS = [
    r"^add\s+",
    r"^i want\s+",
    r"^i want to have\s+",
    r"^can i get\s+",
    r"^can i have\s+",
    r"^give me\s+",
    r"^let me get\s+",
    r"^make it\s+",
]
_WAITING_PREFIX_RES = [re.compile(pat, re.IGNORECASE) for pat in _WAITING_PREFIX_PATTERNS]


@dataclass(frozen=True, slots=True)
class PreprocessedText:
    raw_text: str
    cleaned_text: str
    normalized_text: str


def basic_cleanup(text: Optional[str]) -> str:
    if not text:
        return ""

    s = unicodedata.normalize("NFKC", text)

    for ch in _INVISIBLE_CHARS:
        s = s.replace(ch, "")

    s = _RE_WS.sub(" ", s).strip()
    if not s:
        return ""

    s = _RE_SPACE_BEFORE_PUNCT.sub(r"\1", s)
    s = _RE_MULTI_PUNCT.sub(r"\1", s)
    s = _RE_ELONGATION.sub(r"\1\1", s)
    return s


def _replace_punct_words(text: str) -> str:
    s = text
    for key in sorted(_PUNCT_WORDS.keys(), key=len, reverse=True):
        s = re.sub(rf"\b{re.escape(key)}\b", _PUNCT_WORDS[key], s, flags=re.IGNORECASE)
    return s


def clean_stt_noise(text: Optional[str]) -> str:
    if not text:
        return ""

    s = text.strip()
    if not s:
        return ""

    s = _RE_ASR_BRACKETS.sub(" ", s)
    s = _replace_punct_words(s)
    s = _RE_FILLERS.sub(" ", s)
    s = _RE_DOTS.sub(".", s)
    s = _RE_WS.sub(" ", s).strip()

    if not any(ch.isalnum() for ch in s):
        return ""

    return s


def normalize_text(text: str) -> str:
    if not text:
        return ""

    value = text.lower().strip()
    value = value.translate(_TRANSLATION_TABLE)
    value = _RE_WS.sub(" ", value)
    return value.strip()


def normalize_waiting_value(text: str) -> str:
    if not text:
        return ""

    value = text.strip().lower()
    for pattern in _WAITING_PREFIX_RES:
        value = pattern.sub("", value)

    return value.strip()


def preprocess_turn_text(raw_text: Optional[str]) -> PreprocessedText:
    raw = raw_text or ""
    base_cleaned = basic_cleanup(raw)
    stt_cleaned = clean_stt_noise(base_cleaned)
    normalized = normalize_text(stt_cleaned)
    return PreprocessedText(
        raw_text=raw,
        cleaned_text=stt_cleaned,
        normalized_text=normalized,
    )