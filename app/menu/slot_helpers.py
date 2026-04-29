# app/menu/slot_helpers.py
# Compatibility wrapper — logic now lives in app.nlu.matching.
from app.nlu.matching.index import (  # noqa: F401
    first_slot_value,
    slot_values,
)
