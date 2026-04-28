from __future__ import annotations

from app.intent.confirmation_utils import is_affirmation, is_denial
from app.nlu.control_phrase_lexicon import DEFAULT_LEXICON as _LEXICON
from app.nlu.intent_resolution.intent import Intent
from app.nlu.nlu_result import NLUResult
from app.nlu.query_normalization.text_preprocessor import normalize_text

_LEADING_FILLERS: tuple[str, ...] = (
    "well",
    "so",
    "just",
    "please",
    "uh",
    "um",
    "hmm",
)
_TRAILING_FILLERS: tuple[str, ...] = (
    "please",
    "thanks",
    "thank you",
)
_LEADING_NEGATION_PHRASES: tuple[str, ...] = (
    "no ",
    "nope ",
    "nah ",
)


def _signal_candidates(text: str) -> set[str]:
    normalized = normalize_text(text)
    if not normalized:
        return set()

    candidates: set[str] = {normalized}
    queue: list[str] = [normalized]
    seen: set[str] = set()

    while queue:
        value = queue.pop()
        if not value or value in seen:
            continue
        seen.add(value)
        candidates.add(value)

        for phrase in _LEADING_FILLERS:
            prefix = f"{phrase} "
            if value.startswith(prefix):
                queue.append(value[len(prefix):].strip())

        for prefix in _LEADING_NEGATION_PHRASES:
            if value.startswith(prefix):
                queue.append(value[len(prefix):].strip())

        for phrase in _TRAILING_FILLERS:
            suffix = f" {phrase}"
            if value.endswith(suffix):
                queue.append(value[: -len(suffix)].strip())

        if " and " in value:
            parts = [part.strip() for part in value.split(" and ") if part.strip()]
            queue.extend(parts)
            if len(parts) >= 2 and len(set(parts)) == 1:
                queue.append(parts[0])

    return candidates


def is_affirm_like_response(
    intent: Intent | None,
    text: str,
    *,
    nlu_result: NLUResult | None = None,
    expect_confirmation: bool = False,
) -> bool:
    if is_affirmation(
        nlu_result,
        text,
        resolved_intent=intent,
        expect_confirmation=expect_confirmation,
    ):
        return True
    return _LEXICON.is_affirm(text)


def is_deny_like_response(
    intent: Intent | None,
    text: str,
    *,
    nlu_result: NLUResult | None = None,
    expect_confirmation: bool = False,
) -> bool:
    if is_denial(
        nlu_result,
        text,
        resolved_intent=intent,
        expect_confirmation=expect_confirmation,
    ):
        return True
    return _LEXICON.is_deny(text)
