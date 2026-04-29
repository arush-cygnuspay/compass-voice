# app/utils/deterministic_matcher.py
# Compatibility wrapper — logic now lives in app.nlu.matching.
from app.nlu.matching.matcher import (  # noqa: F401
    _looks_like_skip_choice_answer,
    exact_match,
    resolve_choice,
    token_match,
)
