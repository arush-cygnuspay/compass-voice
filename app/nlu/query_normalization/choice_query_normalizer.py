# app/nlu/query_normalization/choice_query_normalizer.py
import re

from app.nlu.intent_patterns.add_item import CHOICE_FILLER_PAT
from app.nlu.intent_patterns.noise import DISCOURSE_NOISE_PAT
from app.nlu.query_normalization.base import basic_cleanup
from app.nlu.query_normalization.query_normalizer import QueryNormalizer
from app.nlu.intent_resolution.intent import Intent


class ChoiceQueryNormalizer(QueryNormalizer):
    """
    Normalizes text when resolving a SIDE / MODIFIER / SIZE choice.
    """

    def normalize(self, text: str, intent: Intent) -> str:
        cleaned = basic_cleanup(text)

        cleaned = CHOICE_FILLER_PAT.sub("", cleaned)
        cleaned = DISCOURSE_NOISE_PAT.sub("", cleaned)

        # Optional, ONLY if needed later
        cleaned = CHOICE_NOISE_PAT.sub("", cleaned)

        return re.sub(r"\s+", " ", cleaned).strip()
