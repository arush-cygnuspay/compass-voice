# app/nlu/intent_resolution/intent_result.py

from dataclasses import dataclass

from app.nlu.intent_resolution.intent import Intent


@dataclass(frozen=True)
class IntentResult:
    """
    Result of pure linguistic intent resolution.

    NOTE:
    - No session
    - No state
    - No entities
    """
    intent: Intent
    raw_text: str
