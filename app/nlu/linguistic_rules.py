from __future__ import annotations

from app.intent.confirmation_utils import is_affirmation, is_denial
from app.nlu.intent_resolution.intent import Intent
from app.nlu.nlu_result import NLUResult
from app.nlu.query_normalization.text_preprocessor import normalize_text

AFFIRM_WORDS: set[str] = {
    "yes",
    "yeah",
    "yep",
    "yup",
    "correct",
    "thats right",
    "that is right",
    "right",
    "okay",
    "ok",
    "sure",
    "sounds good",
    "mmhmm",
    "mhm",
    "fine",
    "go ahead",
    "please do",
    "do it",
    "confirm",
    "confirm it",
    "that is correct",
    "thats correct",
    "yes it is",
    "yeah its correct",
    "proceed",
    "continue",
    "place it",
    "continue to payment",
    "continue to checkout",
    "checkout",
}

DENY_WORDS: set[str] = {
    "no",
    "nope",
    "nah",
    "wrong",
    "incorrect",
    "not correct",
    "negative",
    "thats wrong",
    "that is wrong",
    "thats not right",
    "that is not right",
    "not now",
    "dont",
    "do not",
    "stop",
    "cancel",
    "leave it",
    "change it",
    "remove something",
    "go back",
}

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
    return any(candidate in AFFIRM_WORDS for candidate in _signal_candidates(text))


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
    return any(candidate in DENY_WORDS for candidate in _signal_candidates(text))
