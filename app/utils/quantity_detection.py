# app/utils/quantity_detection.py
# Compatibility wrapper — logic now lives in app.nlu.matching.
from app.nlu.matching.quantity_parser import (  # noqa: F401
    INCREMENTAL_PATTERNS,
    NUMBER_WORDS,
    SPECIAL_QUANTITIES,
    UNIT_PATTERN,
    UNIT_WORDS,
    VAGUE_PATTERNS,
    WEIGHT_UNIT_TO_OUNCES,
    detect_quantity,
    extract_leading_quantity_phrase,
    extract_weight_quantity,
    normalize_quantity,
)
