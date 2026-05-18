"""NLU resolution + intent confidence gating + state-gated allow-list.

Owns the inline NLU pipeline that previously lived at the top of
``TurnEngine.process_turn``: text preprocessing, intent classification,
slot extraction, confidence gating, and the delivery / waiting-state
allowed-control-intent filter. Returns a typed ``NluResolution`` that
``process_turn`` consumes.

Behavior moved verbatim from ``turn_engine.py``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from app.ml.intent.inference_intent import IntentBundle
from app.ml.slot.inference_slot import SlotBundle
from app.nlu.intent_resolution.intent import Intent
from app.nlu.intent_resolution.intent_result import IntentResult
from app.nlu.nlu_result import NLUResult
from app.nlu.nlu_resolver import resolve_nlu
from app.nlu.query_normalization.text_preprocessor import preprocess_turn_text
from app.session.session import Session
from app.state_machine.common.order_type_resolver import OrderTypeResolver
from app.state_machine.flow_sets import (
    DELIVERY_GATING_ALLOWED_CONTROL_INTENTS,
    WAITING_STATE_ALLOWED_CONTROL_INTENTS,
)
from app.config.nlu import get_nlu_config
from app.state_machine.models.conversation_state import ConversationState

# Sourced from config — no direct os.getenv at module level.
INTENT_MIN_CONF: float = get_nlu_config().intent_conf_threshold


@dataclass(frozen=True, slots=True)
class NluResolution:
    cleaned_text: str
    normalized_text: str
    nlu: NLUResult
    intent_result: IntentResult
    preprocess_ms: float
    nlu_ms: float


class NluOrchestrator:
    """NLU pipeline owner. Construction takes the bundles + diagnostics
    so resolve() is fully self-contained."""

    def __init__(
        self,
        intent_bundle: IntentBundle,
        slot_bundle: SlotBundle,
        diagnostics: Any,
    ) -> None:
        self.intent_bundle = intent_bundle
        self.slot_bundle = slot_bundle
        self.diagnostics = diagnostics

    def resolve(
        self,
        *,
        session: Session,
        user_text: str,
    ) -> NluResolution:
        """Run preprocessing → NLU → confidence gate → allow-list gate.

        Mutates ``session.conversation_context`` (sets last_nlu) the same
        way the inline code did; preserves timing measurements that
        ``process_turn`` then folds into the trace and diagnostics
        outputs.
        """
        ctx = session.conversation_context

        t0 = time.perf_counter()
        preprocessed = preprocess_turn_text(user_text)
        cleaned_text = preprocessed.cleaned_text
        normalized_text = preprocessed.normalized_text
        preprocess_ms = (time.perf_counter() - t0) * 1000.0

        # Fast-path: WAITING_FOR_ORDER_TYPE has a closed answer set (pickup /
        # delivery).  Run the deterministic resolver before the full ML stack.
        # If it matches, skip intent classification and slot extraction entirely.
        if session.conversation_state == ConversationState.WAITING_FOR_ORDER_TYPE:
            order_match = OrderTypeResolver.resolve(normalized_text)
            if order_match is not None:
                nlu = NLUResult(
                    effective_intent=Intent.UNKNOWN,
                    intent_confidence=0.0,
                    raw_text=cleaned_text,
                    normalized_text=normalized_text,
                    slots=(),
                    slot_model_ran=False,
                    nlu_skipped=True,
                    nlu_skip_reason="order_type_lexical_match",
                )
                ctx.set_last_nlu(user_text=cleaned_text, nlu=nlu)
                return NluResolution(
                    cleaned_text=cleaned_text,
                    normalized_text=normalized_text,
                    nlu=nlu,
                    intent_result=IntentResult(
                        intent=Intent.UNKNOWN,
                        raw_text=normalized_text,
                    ),
                    preprocess_ms=preprocess_ms,
                    nlu_ms=0.0,
                )

        t0 = time.perf_counter()
        nlu = resolve_nlu(
            raw_text=cleaned_text,
            normalized_text=normalized_text,
            state=session.conversation_state,
            pending_action=ctx.pending_action,
            intent_bundle=self.intent_bundle,
            slot_bundle=self.slot_bundle,
        )
        ctx.set_last_nlu(user_text=cleaned_text, nlu=nlu)
        nlu_ms = (time.perf_counter() - t0) * 1000.0

        detected_intent = (
            nlu.effective_intent
            if nlu.intent_confidence >= INTENT_MIN_CONF
            else Intent.UNKNOWN
        )
        intent_result = IntentResult(
            intent=detected_intent,
            raw_text=nlu.normalized_text,
        )

        delivery_gating_states = {
            ConversationState.WAITING_FOR_DELIVERY_ELIGIBILITY,
            ConversationState.WAITING_FOR_DELIVERY_ADDRESS_COLLECTION,
        }

        generic_waiting_states = {
            ConversationState.WAITING_FOR_SIDE,
            ConversationState.WAITING_FOR_SIDE_SIZE,
            ConversationState.WAITING_FOR_MODIFIER,
            ConversationState.WAITING_FOR_SIZE,
            ConversationState.WAITING_FOR_QUANTITY,
        }

        if session.conversation_state in delivery_gating_states:
            allowed_control_intents = DELIVERY_GATING_ALLOWED_CONTROL_INTENTS
        elif session.conversation_state in generic_waiting_states:
            allowed_control_intents = WAITING_STATE_ALLOWED_CONTROL_INTENTS
        else:
            allowed_control_intents = set()

        if (
            session.conversation_state in delivery_gating_states | generic_waiting_states
            and intent_result.intent not in allowed_control_intents
        ):
            intent_result = IntentResult(
                intent=Intent.UNKNOWN,
                raw_text=nlu.normalized_text,
            )

        return NluResolution(
            cleaned_text=cleaned_text,
            normalized_text=normalized_text,
            nlu=nlu,
            intent_result=intent_result,
            preprocess_ms=preprocess_ms,
            nlu_ms=nlu_ms,
        )
