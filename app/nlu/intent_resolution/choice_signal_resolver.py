# app/nlu/choice_signal_resolver.py
from app.nlu.intent_patterns.common import *
from app.nlu.intent_patterns.ask_options import *
from app.nlu.intent_resolution.choice_signals import ChoiceSignal


def resolve_choice_signal(text: str) -> ChoiceSignal:
    if not text:
        return ChoiceSignal.NONE

    # 1️⃣ Cancel has highest priority
    if NO_CANCEL_PAT.search(text):
        return ChoiceSignal.CANCEL

    # 2️⃣ Asking for options
    if any(
        pat.search(text)
        for pat in (
            ASK_OPTIONS_CORE_PAT,
            ASK_OPTIONS_FOR_ENTITY_PAT,
            ASK_OPTIONS_FOLLOWUP_PAT,
            ASK_OPTIONS_CASUAL_PAT,
        )
    ):
        return ChoiceSignal.ASK_OPTIONS

    # 3️⃣ Deny / skip
    if NO_STRONG_PAT.search(text) or NO_NEGATION_PAT.search(text):
        return ChoiceSignal.DENY

    # 4️⃣ Confirm (rare in choice states, but valid)
    if YES_STRONG_PAT.search(text):
        return ChoiceSignal.CONFIRM

    return ChoiceSignal.NONE
