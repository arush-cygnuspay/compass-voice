from __future__ import annotations

from app.nlu.intent_resolution.intent import Intent
from app.nlu.nlu_result import IntentCandidate, SlotValue
from app.nlu.semantic_repair.gpt_execution_policy import (
    GptExecutionMode,
    GptExecutionPolicy,
    GptPromptBucket,
)
from app.nlu.semantic_repair.gpt_log_record_builder import build_gpt_shadow_jsonl_record
from app.nlu.semantic_repair.gpt_repair_result import GptRepairResult
from app.nlu.semantic_repair.repair_service import LocalTurnAnalysis
from app.state_machine.models.conversation_state import ConversationState


def _candidate(intent: str, confidence: float) -> IntentCandidate:
    return IntentCandidate(
        intent_main="order",
        intent_sub_intent=intent,
        canonical_intent=intent,
        confidence=confidence,
    )


def _decide(**overrides):
    policy = GptExecutionPolicy()
    payload = {
        "state": ConversationState.IDLE,
        "normalized_user_text": "burger",
        "raw_stt_final_text": "burger",
        "local_intent_top_n": (_candidate("add_item", 0.96),),
        "selected_local_intent": Intent.ADD_ITEM,
        "local_intent_confidence": 0.96,
        "local_slots": (),
        "active_pending_item_context": None,
        "available_options_context": (),
        "fallback_count": 0,
        "repeated_prompt_count": 0,
        "previous_turns_summary": (),
        "handler_resolution_status": None,
        "last_response_key": None,
        "duplicate_transcript": False,
    }
    payload.update(overrides)
    return policy.decide(**payload)


def test_high_confidence_exact_option_match_returns_none() -> None:
    decision = _decide(
        state=ConversationState.WAITING_FOR_SIDE,
        normalized_user_text="fries",
        raw_stt_final_text="fries",
        local_intent_confidence=0.95,
        available_options_context=("Fries", "Salad"),
    )
    assert decision.mode == GptExecutionMode.NONE


def test_clean_high_confidence_add_item_returns_shadow() -> None:
    decision = _decide()
    assert decision.mode == GptExecutionMode.SHADOW
    assert decision.prompt_bucket == GptPromptBucket.ADD_ITEM_PLAN


def test_low_confidence_unknown_returns_inline_with_timeout() -> None:
    decision = _decide(
        normalized_user_text="something good",
        raw_stt_final_text="something good",
        selected_local_intent=Intent.UNKNOWN,
        local_intent_confidence=0.21,
        local_intent_top_n=(_candidate("unknown", 0.21), _candidate("checkout", 0.19)),
    )
    assert decision.mode == GptExecutionMode.INLINE_WITH_TIMEOUT


def test_waiting_for_side_unresolved_option_uses_side_selection_bucket() -> None:
    decision = _decide(
        state=ConversationState.WAITING_FOR_SIDE,
        normalized_user_text="curly potato",
        raw_stt_final_text="curly potato",
        selected_local_intent=Intent.UNKNOWN,
        local_intent_confidence=0.2,
        available_options_context=("Fries", "Salad"),
    )
    assert decision.mode == GptExecutionMode.INLINE_WITH_TIMEOUT
    assert decision.prompt_bucket == GptPromptBucket.SIDE_SELECTION


def test_waiting_for_modifier_unresolved_option_uses_modifier_selection_bucket() -> None:
    decision = _decide(
        state=ConversationState.WAITING_FOR_MODIFIER,
        normalized_user_text="pepper jack",
        raw_stt_final_text="pepper jack",
        selected_local_intent=Intent.UNKNOWN,
        local_intent_confidence=0.2,
        available_options_context=("Swiss Cheese", "Mozzarella Cheese"),
    )
    assert decision.mode == GptExecutionMode.INLINE_WITH_TIMEOUT
    assert decision.prompt_bucket == GptPromptBucket.MODIFIER_SELECTION


def test_macarola_cheese_with_mozzarella_option_prefers_inline_modifier_repair() -> None:
    decision = _decide(
        state=ConversationState.WAITING_FOR_MODIFIER,
        normalized_user_text="macarola cheese",
        raw_stt_final_text="macarola cheese",
        selected_local_intent=Intent.UNKNOWN,
        local_intent_confidence=0.18,
        available_options_context=("Mozzarella Cheese", "Swiss Cheese"),
    )
    assert decision.mode == GptExecutionMode.INLINE_WITH_TIMEOUT
    assert decision.prompt_bucket == GptPromptBucket.MODIFIER_SELECTION
    assert "phonetic_option_mismatch" in decision.reason_codes


def test_denial_with_replacement_returns_inline_correction_bucket() -> None:
    decision = _decide(
        state=ConversationState.CONFIRMING_ITEM,
        normalized_user_text="no, chicken burger",
        raw_stt_final_text="no, chicken burger",
        selected_local_intent=Intent.DENY,
        local_intent_confidence=0.88,
        previous_turns_summary=(("bot", "Did you mean cheese burger?"),),
    )
    assert decision.mode == GptExecutionMode.INLINE
    assert decision.prompt_bucket == GptPromptBucket.CORRECTION_OR_DENIAL


def test_repeated_fallback_uses_fallback_recovery_bucket() -> None:
    decision = _decide(
        state=ConversationState.WAITING_FOR_SIDE,
        normalized_user_text="whatever",
        raw_stt_final_text="whatever",
        selected_local_intent=Intent.UNKNOWN,
        local_intent_confidence=0.2,
        fallback_count=3,
        available_options_context=("Fries", "Salad"),
    )
    assert decision.mode == GptExecutionMode.INLINE_WITH_TIMEOUT
    assert decision.prompt_bucket == GptPromptBucket.FALLBACK_RECOVERY


def test_checkout_while_waiting_for_modifier_uses_checkout_bucket() -> None:
    decision = _decide(
        state=ConversationState.WAITING_FOR_MODIFIER,
        normalized_user_text="checkout",
        raw_stt_final_text="checkout",
        selected_local_intent=Intent.UNKNOWN,
        local_intent_confidence=0.35,
        available_options_context=("Swiss Cheese",),
    )
    assert decision.mode == GptExecutionMode.INLINE
    assert decision.prompt_bucket == GptPromptBucket.CHECKOUT_OR_PAYMENT


def test_silence_returns_none() -> None:
    decision = _decide(
        normalized_user_text="",
        raw_stt_final_text="",
        selected_local_intent=Intent.UNKNOWN,
        local_intent_confidence=0.0,
    )
    assert decision.mode == GptExecutionMode.NONE


def test_duplicate_transcript_returns_none() -> None:
    decision = _decide(duplicate_transcript=True)
    assert decision.mode == GptExecutionMode.NONE


def test_top1_top2_close_returns_inline_with_timeout() -> None:
    decision = _decide(
        normalized_user_text="something",
        raw_stt_final_text="something",
        selected_local_intent=Intent.SHOW_CART,
        local_intent_confidence=0.51,
        local_intent_top_n=(_candidate("show_cart", 0.51), _candidate("checkout", 0.45)),
    )
    assert decision.mode == GptExecutionMode.INLINE_WITH_TIMEOUT


def test_slots_present_but_handler_unresolved_returns_inline_with_timeout() -> None:
    decision = _decide(
        state=ConversationState.WAITING_FOR_MODIFIER,
        normalized_user_text="extra cheese",
        raw_stt_final_text="extra cheese",
        selected_local_intent=Intent.ADD_ITEM,
        local_intent_confidence=0.74,
        local_slots=(SlotValue(name="MODIFIER", value="cheese"),),
        handler_resolution_status="unresolved",
    )
    assert decision.mode == GptExecutionMode.INLINE_WITH_TIMEOUT


def test_multi_item_utterance_returns_inline_add_item_plan() -> None:
    decision = _decide(
        normalized_user_text="burger and fries",
        raw_stt_final_text="burger and fries",
        selected_local_intent=Intent.ADD_ITEM,
        local_intent_confidence=0.86,
        local_slots=(SlotValue(name="ITEM", value="Burger"),),
    )
    assert decision.mode == GptExecutionMode.INLINE
    assert decision.prompt_bucket == GptPromptBucket.ADD_ITEM_PLAN


def test_jsonl_record_does_not_include_removed_policy_block() -> None:
    """The 'policy' block was removed from the JSONL record in Phase 2 cleanup.

    GptExecutionPolicy is a Phase 3 untracked file.  The old 'policy' stanza
    was removed from build_gpt_shadow_jsonl_record() as part of Phase 2 audit
    remediation.  This test guards against it being re-introduced.

    The JSONL record MUST contain the canonical Phase 2 sections:
    local, allowed, gpt, final.  It must NOT contain 'policy'.
    """
    # Build analysis without the removed execution_decision / execution_policy_ms fields.
    analysis = LocalTurnAnalysis(
        gpt_repair_eligible=True,
        reason="top1_top2_close",
        candidate_count=1,
        candidates=frozenset({"add_item"}),
        intent_effective=Intent.UNKNOWN.value,
        intent_confidence=0.18,
        intent_candidates=(_candidate("unknown", 0.18),),
        state_before=ConversationState.WAITING_FOR_MODIFIER.value,
        customer_text="macarola cheese",
        normalized_text="macarola cheese",
        slots=(),
        candidate_repair_intents=frozenset({"add_item"}),
        candidate_control_kinds=frozenset({"cancel", "confirm", "deny"}),
        choices=("Mozzarella Cheese",),
        previous_turns=(("bot", "Which cheese would you like?"),),
    )
    record = build_gpt_shadow_jsonl_record(
        analysis=analysis,
        result=GptRepairResult(decision="no_repair"),
        nlu=type("Nlu", (), {"normalized_text": "macarola cheese"})(),
        state_before=ConversationState.WAITING_FOR_MODIFIER.value,
        session_id="s1",
        turn_index=1,
        response_key="ask_for_modifier",
    )
    # 'policy' block must be ABSENT — it was removed in Phase 2 cleanup.
    assert "policy" not in record, (
        "The 'policy' block was removed from JSONL records in Phase 2 audit "
        "remediation and must not be re-introduced."
    )
    # Canonical Phase 2 sections must still be present.
    assert "local" in record
    assert "allowed" in record
    assert "gpt" in record
    assert "final" in record
    assert record["session_id"] == "s1"
    assert record["turn_index"] == 1
