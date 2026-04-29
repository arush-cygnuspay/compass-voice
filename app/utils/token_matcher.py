# app/utils/token_matcher.py
# Compatibility wrapper — logic now lives in app.nlu.matching.
from app.nlu.matching.normalization import (  # noqa: F401
    _WEAK_TOKENS,
    _canonicalize_token,
    tokenize,
)
from app.nlu.matching.matcher import (  # noqa: F401
    is_controlled_partial_match,
    is_strong_token_match,
    token_overlap_score,
)
