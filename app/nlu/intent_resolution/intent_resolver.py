# app/nlu/intent_resolution/intent_resolver.py
from __future__ import annotations

import os

from app.ml.intent.inference_intent import IntentBundle, predict_intent
from app.nlu.intent_resolution.intent import Intent
from app.nlu.intent_resolution.intent_mapping import SUB_INTENT_TO_INTENT
from app.nlu.intent_resolution.intent_result import IntentResult
from app.state_machine.conversation_state import ConversationState


CONFIDENCE_THRESHOLD = float(os.getenv("COMPASS_INTENT_CONF_THRESHOLD", "0.55"))


def predict_intent_labels(
    text: str,
    bundle: IntentBundle,
) -> tuple[str | None, str | None, float, float]:
    results = predict_intent(
        texts=text,
        bundle=bundle,
        max_length=64,
    )
    if not results:
        return None, None, 0.0, 0.0

    result = results[0]
    main = result.get("pred_main_intent")
    sub = result.get("pred_sub_intent")

    confidence_main = float(result.get("confidence_main", 0.0) or 0.0)
    confidence_sub = float(result.get("confidence_sub", 0.0) or 0.0)

    main = main.strip() if isinstance(main, str) else None
    sub = sub.strip() if isinstance(sub, str) else None

    return main, sub, confidence_main, confidence_sub


def resolve_intent(
    text: str,
    state: ConversationState,
    bundle: IntentBundle,
) -> IntentResult:
    if not text:
        return IntentResult(Intent.UNKNOWN, "")

    _main, sub, _confidence_main, confidence_sub = predict_intent_labels(text, bundle)

    if not sub or confidence_sub < CONFIDENCE_THRESHOLD:
        return IntentResult(Intent.UNKNOWN, text)

    return IntentResult(SUB_INTENT_TO_INTENT.get(sub, Intent.UNKNOWN), text)