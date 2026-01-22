# app/nlu/query_normalization/noise_cleaner.py

from app.nlu.intent_patterns.noise import *


def clean_stt_noise(text: str) -> str:
    """
    Removes STT conversational noise WITHOUT changing semantics.

    IMPORTANT:
    - Runs BEFORE intent detection
    - Does NOT remove yes/no
    - Does NOT remove goodbye/greeting (yet)
    - Safe for future extension
    """
    cleaned = text

    # Remove fillers
    cleaned = FILLER_NOISE_PAT.sub(" ", cleaned)

    # Remove discourse markers
    cleaned = DISCOURSE_NOISE_PAT.sub(" ", cleaned)

    # Remove politeness
    cleaned = POLITENESS_NOISE_PAT.sub(" ", cleaned)

    # Remove item fluff
    cleaned = ITEM_NOISE_PAT.sub(" ", cleaned)

    # Remove prefix noise (carefully)
    cleaned = PREFIX_NOISE_PAT.sub("", cleaned)

    # Remove trailing fluff
    cleaned = TRAILING_NOISE_PAT.sub("", cleaned)

    # Normalize spacing
    cleaned = " ".join(cleaned.split())

    return cleaned
