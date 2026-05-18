# app/nlu/intent_resolution/intent_resolver.py
from __future__ import annotations

import os
from typing import TYPE_CHECKING

from app.ml.intent.inference_intent import IntentBundle, predict_intent
from app.nlu.intent_resolution.intent import Intent
from app.nlu.intent_resolution.intent_mapping import SUB_INTENT_TO_INTENT
from app.nlu.intent_resolution.intent_result import IntentResult
from app.nlu.nlu_result import IntentCandidate
from app.state_machine.models.conversation_state import ConversationState


CONFIDENCE_THRESHOLD = float(os.getenv("COMPASS_INTENT_CONF_THRESHOLD", "0.55"))


def predict_intent_labels(
    text: str,
    bundle: IntentBundle,
    top_k: int = 4,
) -> tuple[str | None, str | None, float, float, tuple[IntentCandidate, ...]]:
    """Return (main, sub, conf_main, conf_sub, top_k_candidates).

    The 5th element is a tuple of IntentCandidate objects sorted by confidence.
    Pass top_k=1 to skip candidate extraction (returns empty tuple).
    """
    results = predict_intent(
        texts=text,
        bundle=bundle,
        max_length=64,
        top_k=max(top_k, 1),
    )
    if not results:
        return None, None, 0.0, 0.0, ()

    result = results[0]
    main = result.get("pred_main_intent")
    sub = result.get("pred_sub_intent")

    confidence_main = float(result.get("confidence_main", 0.0) or 0.0)
    confidence_sub = float(result.get("confidence_sub", 0.0) or 0.0)

    main = main.strip() if isinstance(main, str) else None
    sub = sub.strip() if isinstance(sub, str) else None

    candidates: tuple[IntentCandidate, ...] = ()
    if top_k > 1:
        top_k_subs = result.get("top_k_sub_intents") or []
        main_str = main or ""
        candidates = tuple(
            IntentCandidate(
                intent_main=main_str,
                intent_sub_intent=(item.get("sub_intent") or "").strip(),
                canonical_intent=SUB_INTENT_TO_INTENT.get(
                    (item.get("sub_intent") or "").strip(), Intent.UNKNOWN
                ).value,
                confidence=float(item.get("confidence", 0.0)),
                source="model_sub",
            )
            for item in top_k_subs
            if item.get("sub_intent")
        )

    return main, sub, confidence_main, confidence_sub, candidates


def resolve_intent(
    text: str,
    state: ConversationState,
    bundle: IntentBundle,
) -> IntentResult:
    if not text:
        return IntentResult(Intent.UNKNOWN, "")

    _main, sub, _confidence_main, confidence_sub, _ = predict_intent_labels(text, bundle, top_k=1)

    if not sub or confidence_sub < CONFIDENCE_THRESHOLD:
        return IntentResult(Intent.UNKNOWN, text)

    return IntentResult(SUB_INTENT_TO_INTENT.get(sub, Intent.UNKNOWN), text)