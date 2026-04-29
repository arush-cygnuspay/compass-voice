from __future__ import annotations

from app.nlu.intent_resolution.confirmation_resolver import resolve_confirmation_decision
from app.nlu.intent_resolution.intent import Intent
from app.nlu.linguistic_rules import (
    _signal_candidates,
    is_affirm_like_response,
    is_deny_like_response,
)
from app.nlu.nlu_result import NLUResult

AFFIRM_INTENTS: set[Intent] = {Intent.AFFIRM, Intent.CONFIRM}
DENY_INTENTS: set[Intent] = {Intent.DENY}
OPTIONS_INTENTS: set[Intent] = {Intent.ASK_OPTIONS}

OPTIONAL_SKIP_WORDS: set[str] = {
    "none",
    "nothing",
    "skip",
    "skip it",
    "no thanks",
    "nothing else",
    "no more",
    "that is all",
    "thats all",
    "im good",
    "i am good",
}

DONE_WORDS: set[str] = {
    "done",
    "i am done",
    "im done",
    "done and done",
    "no done",
    "no no more",
    "that is all",
    "thats all",
    "that is it",
    "thats it",
    "finished",
    "i am finished",
    "im finished",
    "no more",
    "nothing else",
    "i am good",
    "im good",
    "we are good",
    "were good",
    "all good",
    "that should do it",
    "that will do it",
    "thatll do it",
    "that is enough",
    "thats enough",
    "okay thats good",
    "okay thats good thanks",
    "sounds good",
    "good",
}

OPTIONS_WORDS: set[str] = {
    "options",
    "what are the options",
    "what are my options",
    "what can i choose",
    "tell me the choices",
    "what do you have",
    "list options",
    "available toppings",
    "what else",
    "what else do you have",
    "what else you got",
    "more options",
    "other options",
    "show me more",
    "any others",
}

REPEAT_WORDS: set[str] = {
    "repeat",
    "repeat that",
    "say that again",
    "come again",
    "what was that",
    "can you repeat that",
}

HELP_WORDS: set[str] = {
    "help",
    "i need help",
    "what can i say",
    "what should i say",
}


def is_optional_skip_response(intent: Intent | None, text: str) -> bool:
    if intent in DENY_INTENTS:
        return True
    return any(candidate in OPTIONAL_SKIP_WORDS for candidate in _signal_candidates(text))


def is_done_like_response(intent: Intent | None, text: str) -> bool:
    if intent in {Intent.END_ADDING, Intent.CHECKOUT, Intent.FINISH_ORDER, Intent.CONFIRM_ORDER}:
        return True
    return any(candidate in DONE_WORDS for candidate in _signal_candidates(text))


def is_options_like_response(intent: Intent | None, text: str) -> bool:
    if intent in OPTIONS_INTENTS:
        return True
    return any(candidate in OPTIONS_WORDS for candidate in _signal_candidates(text))


def is_repeat_like_response(text: str) -> bool:
    return any(candidate in REPEAT_WORDS for candidate in _signal_candidates(text))


def is_help_like_response(text: str) -> bool:
    return any(candidate in HELP_WORDS for candidate in _signal_candidates(text))


def is_confirmation_accept_response(
    intent: Intent | None,
    text: str,
    *,
    nlu_result: NLUResult | None = None,
) -> bool:
    return is_affirm_like_response(
        intent,
        text,
        nlu_result=nlu_result,
        expect_confirmation=True,
    )


def is_confirmation_reject_response(
    intent: Intent | None,
    text: str,
    *,
    nlu_result: NLUResult | None = None,
) -> bool:
    return resolve_confirmation_decision(
        nlu_result,
        text,
        resolved_intent=intent,
        expect_confirmation=True,
    ) in {"deny", "cancel"}
