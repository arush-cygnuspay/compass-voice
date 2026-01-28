# app/nlu/choice_signals/resolver.py
from app.nlu.choice_signals.choice_signals import ChoiceSignal
from app.nlu.intent_patterns.common import *
from app.nlu.intent_patterns.ask_options import *


def resolve_choice_signal(text: str) -> ChoiceSignal:
    """
    Interprets user input *within a choice-selection step*.
    This is NOT global intent detection.
    """

    if not text:
        return ChoiceSignal.NONE

    text = text.strip().lower()

    # 1️⃣ Cancel (absolute priority)
    if NO_CANCEL_PAT.search(text):
        return ChoiceSignal.CANCEL

    # 2️⃣ High-recall options (voice-friendly)
    if BASIC_OPTIONS_PAT.search(text):
        return ChoiceSignal.ASK_OPTIONS

    # 3️⃣ Precise patterns (semantic)
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

    # 4️⃣ Deny / skip
    if NO_STRONG_PAT.search(text) or NO_NEGATION_PAT.search(text):
        return ChoiceSignal.DENY

    # 5️⃣ Confirm
    if YES_STRONG_PAT.search(text):
        return ChoiceSignal.CONFIRM

    return ChoiceSignal.NONE
