# app/utils/item_matching.py
# Compatibility wrapper — logic now lives in app.nlu.matching.
from app.nlu.matching.scorer import (  # noqa: F401
    _ngrams,
    _tokens_from_normalized,
    score_item,
    score_item_normalized,
)
