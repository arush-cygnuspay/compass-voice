# app/nlu/nlu_resolver.py
from __future__ import annotations

import os
import re
from typing import Optional

from app.core.pending_action import PendingAction
from app.ml.intent.inference_intent import IntentBundle
from app.ml.slot.inference_slot import SlotBundle
from app.nlu.intent_resolution.intent import Intent
from app.nlu.intent_resolution.intent_mapping import SUB_INTENT_TO_INTENT
from app.nlu.intent_resolution.intent_resolver import predict_intent_labels
from app.nlu.nlu_result import NLUResult, SlotValue
from app.nlu.slot_resolution.slot_resolver import predict_slots
from app.state_machine.models.conversation_state import ConversationState
from app.utils.quantity_detection import normalize_quantity


SLOTS_ENABLED = os.getenv("COMPASS_SLOTS_ENABLED", "1") != "0"

WAITING_STATES = {
    ConversationState.WAITING_FOR_SIDE,
    ConversationState.WAITING_FOR_MODIFIER,
    ConversationState.WAITING_FOR_SIZE,
    ConversationState.WAITING_FOR_SIDE_SIZE,
    ConversationState.WAITING_FOR_QUANTITY,
}

SLOT_SUBINTENTS: set[str] = {
    "add_item",
    "remove_item",
    "modify_item",
    "ask_item_info",
    "ask_price",
    "availability_query",
    "ask_options",
    "browse_category",
    "browse_menu",
}

QUANTITY_PATTERNS = (
    r"\bhalf dozen\b",
    r"\ba dozen\b",
    r"\b(?:\d+|a|an|single|couple|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+dozen\b",
    r"\b(?:\d+|a|an|single|couple|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+(?:pieces|piece|pcs|pc|orders|order)\b",
    r"\b\d+\b",
    r"\b(?:a|an|single|couple|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b",
)

SIZE_WORDS = {
    "small",
    "medium",
    "large",
    "regular",
    "mini",
    "xl",
    "extra large",
}

CONTROL_PHRASES = {
    "cancel",
    "never mind",
    "nevermind",
    "stop",
    "checkout",
    "show cart",
    "show total",
}

YES_PATTERNS = {
    "yes", "yeah", "yep", "yup", "correct", "right", "sure", "okay", "ok",
    "go ahead", "do it", "confirm", "please do", "that's right", "thats right",
}

NO_PATTERNS = {
    "no", "nope", "nah", "negative", "don't", "do not", "not now",
    "keep it", "continue", "no thanks", "that's wrong", "thats wrong",
}


def _should_run_slots(*, sub_intent: Optional[str], state: ConversationState) -> bool:
    if not SLOTS_ENABLED:
        return False

    if state in WAITING_STATES:
        return True

    return bool(sub_intent) and sub_intent in SLOT_SUBINTENTS


def _dicts_to_slotvalues(slot_dicts: list[dict]) -> tuple[SlotValue, ...]:
    return tuple(
        SlotValue(
            name=str(slot.get("slot", "")),
            value=slot.get("value"),
            raw=slot.get("value"),
            start=slot.get("start"),
            end=slot.get("end"),
            confidence=None,
        )
        for slot in slot_dicts
    )


def _extract_quantity_rule(normalized: str) -> tuple[SlotValue, ...]:
    for pattern in QUANTITY_PATTERNS:
        match = re.search(pattern, normalized)
        if not match:
            continue

        raw = match.group(0)
        value = normalize_quantity(raw)
        if value is None:
            continue

        return (
            SlotValue(
                name="QUANTITY",
                value=value,
                raw=raw,
                start=match.start(),
                end=match.end(),
                confidence=1.0,
            ),
        )

    return ()


def _extract_size_rule(normalized: str) -> tuple[SlotValue, ...]:
    for size in sorted(SIZE_WORDS, key=len, reverse=True):
        match = re.search(rf"\b{re.escape(size)}\b", normalized)
        if match:
            return (
                SlotValue(
                    name="SIZE",
                    value=size,
                    raw=size,
                    start=match.start(),
                    end=match.end(),
                    confidence=1.0,
                ),
            )
    return ()


def _extract_confirmation_intent(normalized: str) -> Intent | None:
    if normalized in YES_PATTERNS:
        return Intent.CONFIRM
    if normalized in NO_PATTERNS:
        return Intent.DENY
    return None


def _build_rule_only_result(
    *,
    raw_text: str,
    normalized_text: str,
    effective_intent: Intent,
    model_main_intent: str | None,
    model_sub_intent: str | None,
) -> NLUResult:
    return NLUResult(
        effective_intent=effective_intent,
        intent_confidence=1.0,
        raw_text=raw_text,
        normalized_text=normalized_text,
        model_main_intent=model_main_intent,
        model_sub_intent=model_sub_intent,
        slots=(),
        slot_model_ran=False,
    )


def _extract_waiting_state_slots(
    normalized_text: str,
    state: ConversationState,
) -> tuple[SlotValue, ...]:
    if state == ConversationState.WAITING_FOR_QUANTITY:
        return _extract_quantity_rule(normalized_text)

    if state in {
        ConversationState.WAITING_FOR_SIZE,
        ConversationState.WAITING_FOR_SIDE_SIZE,
    }:
        return _extract_size_rule(normalized_text)

    return ()


def _looks_like_control_utterance(normalized_text: str) -> bool:
    return any(phrase in normalized_text for phrase in CONTROL_PHRASES)


def _build_waiting_result(
    *,
    raw_text: str,
    normalized_text: str,
    slots: tuple[SlotValue, ...],
    slot_model_ran: bool,
    model_main_intent: str | None,
    model_sub_intent: str | None,
) -> NLUResult:
    return NLUResult(
        effective_intent=Intent.UNKNOWN,
        intent_confidence=1.0,
        raw_text=raw_text,
        normalized_text=normalized_text,
        model_main_intent=model_main_intent,
        model_sub_intent=model_sub_intent,
        slots=slots,
        slot_model_ran=slot_model_ran,
    )


def _resolve_waiting_state(
    *,
    raw_text: str,
    normalized_text: str,
    state: ConversationState,
    model_main_intent: str | None,
    model_sub_intent: str | None,
    slot_bundle: SlotBundle,
) -> NLUResult | None:
    if state not in WAITING_STATES:
        return None

    fast_slots = _extract_waiting_state_slots(normalized_text, state)
    if fast_slots:
        return _build_waiting_result(
            raw_text=raw_text,
            normalized_text=normalized_text,
            slots=fast_slots,
            slot_model_ran=False,
            model_main_intent=model_main_intent,
            model_sub_intent=model_sub_intent,
        )

    if not _looks_like_control_utterance(normalized_text):
        try:
            slot_result = predict_slots(normalized_text, bundle=slot_bundle)[0]
            slot_dicts = slot_result.get("slots", []) or []
            slots = _dicts_to_slotvalues(slot_dicts)

            if slots:
                return _build_waiting_result(
                    raw_text=raw_text,
                    normalized_text=normalized_text,
                    slots=slots,
                    slot_model_ran=True,
                    model_main_intent=model_main_intent,
                    model_sub_intent=model_sub_intent,
                )
        except Exception:
            pass

    return None


def resolve_nlu(
    *,
    raw_text: str,
    normalized_text: str,
    state: ConversationState,
    pending_action: PendingAction | None,
    intent_bundle: IntentBundle,
    slot_bundle: SlotBundle,
) -> NLUResult:
    if not raw_text or not normalized_text:
        return NLUResult(
            effective_intent=Intent.UNKNOWN,
            intent_confidence=0.0,
            raw_text=raw_text or "",
            normalized_text=normalized_text or "",
            model_main_intent=None,
            model_sub_intent=None,
            slots=(),
            slot_model_ran=False,
        )

    model_main_label, model_sub_label, confidence_main, confidence_sub = predict_intent_labels(
        normalized_text,
        intent_bundle,
    )

    if not model_sub_label:
        model_predicted_intent = Intent.UNKNOWN
        model_predicted_conf = 0.0
    else:
        model_predicted_intent = SUB_INTENT_TO_INTENT.get(model_sub_label, Intent.UNKNOWN)
        model_predicted_conf = float(confidence_sub)

    if state in {
        ConversationState.CANCELLATION_CONFIRMATION,
        ConversationState.CONFIRMING_ITEM,
        ConversationState.CONFIRMING_ORDER,
    }:
        confirm_intent = _extract_confirmation_intent(normalized_text)
        if confirm_intent is not None:
            return _build_rule_only_result(
                raw_text=raw_text,
                normalized_text=normalized_text,
                effective_intent=confirm_intent,
                model_main_intent=model_main_label,
                model_sub_intent=model_sub_label,
            )

    waiting_result = _resolve_waiting_state(
        raw_text=raw_text,
        normalized_text=normalized_text,
        state=state,
        model_main_intent=model_main_label,
        model_sub_intent=model_sub_label,
        slot_bundle=slot_bundle,
    )
    if waiting_result is not None:
        return waiting_result

    slot_model_ran = _should_run_slots(sub_intent=model_sub_label, state=state)
    slots: tuple[SlotValue, ...] = ()

    if slot_model_ran:
        slot_result = predict_slots(normalized_text, bundle=slot_bundle)[0]
        slot_dicts = slot_result.get("slots", []) or []
        slots = _dicts_to_slotvalues(slot_dicts)

    return NLUResult(
        effective_intent=model_predicted_intent,
        intent_confidence=model_predicted_conf,
        raw_text=raw_text,
        normalized_text=normalized_text,
        model_main_intent=model_main_label,
        model_sub_intent=model_sub_label,
        slots=slots,
        slot_model_ran=slot_model_ran,
    )
