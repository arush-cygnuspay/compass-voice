from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import StrEnum
from typing import Iterable

from app.nlu.intent_resolution.intent import Intent
from app.nlu.nlu_result import IntentCandidate, SlotValue
from app.nlu.semantic_repair.prompt_builder import get_candidates
from app.state_machine.models.conversation_state import ConversationState


_WAITING_SIDE_STATES: frozenset[ConversationState] = frozenset({
    ConversationState.WAITING_FOR_SIDE,
    ConversationState.WAITING_FOR_SIDE_SIZE,
})
_WAITING_MODIFIER_STATES: frozenset[ConversationState] = frozenset({
    ConversationState.WAITING_FOR_MODIFIER,
})
_WAITING_SELECTION_STATES: frozenset[ConversationState] = (
    _WAITING_SIDE_STATES
    | _WAITING_MODIFIER_STATES
    | frozenset({
        ConversationState.WAITING_FOR_SIZE,
        ConversationState.WAITING_FOR_QUANTITY,
    })
)
_NO_SPEECH_MARKERS: frozenset[str] = frozenset({
    "",
    "[silence]",
    "<silence>",
    "[no-speech]",
    "<no-speech>",
    "[inaudible]",
})
_YES_NO_INTENTS: frozenset[Intent] = frozenset({
    Intent.CONFIRM,
    Intent.DENY,
})
_ORDER_EXIT_PHRASES: tuple[str, ...] = (
    "checkout",
    "check out",
    "done",
    "that's all",
    "thats all",
    "finish",
    "pay",
    "payment",
)
_DENIAL_PREFIXES: tuple[str, ...] = (
    "no ",
    "no,",
    "nope",
    "nah",
    "not ",
)
_MULTI_ITEM_SEPARATORS: tuple[str, ...] = (" and ", ",", " with a side of ", " plus ")
_SIDE_OR_MODIFIER_HINTS: tuple[str, ...] = (
    "with ",
    "extra ",
    "no ",
    "without ",
    "add ",
)


class GptExecutionMode(StrEnum):
    NONE = "none"
    SHADOW = "shadow"
    INLINE = "inline"
    INLINE_WITH_TIMEOUT = "inline_with_timeout"
    REPAIR_ONLY = "repair_only"


class GptPromptBucket(StrEnum):
    ADD_ITEM_PLAN = "add_item_plan"
    SIDE_SELECTION = "side_selection"
    MODIFIER_SELECTION = "modifier_selection"
    CORRECTION_OR_DENIAL = "correction_or_denial"
    CHECKOUT_OR_PAYMENT = "checkout_or_payment"
    CANCEL_OR_REMOVE = "cancel_or_remove"
    GENERAL_STATE_REPAIR = "general_state_repair"
    FALLBACK_RECOVERY = "fallback_recovery"


@dataclass(frozen=True, slots=True)
class GptTopIntent:
    intent: str
    confidence: float


@dataclass(frozen=True, slots=True)
class GptExecutionDecision:
    mode: GptExecutionMode
    prompt_bucket: GptPromptBucket
    reason_codes: tuple[str, ...]
    allowed_intents: tuple[str, ...]
    top_intents: tuple[GptTopIntent, ...]
    state: str
    confidence: float
    timeout_ms: int
    include_previous_turns_count: int
    include_constrained_options: bool
    include_pending_item_context: bool
    include_menu_candidates: bool
    should_validate: bool
    should_apply_result: bool
    should_log_shadow_result: bool

    @property
    def reason_summary(self) -> str:
        return "|".join(self.reason_codes)


class GptExecutionPolicy:
    def __init__(
        self,
        *,
        low_confidence_threshold: float = 0.72,
        close_confidence_gap_threshold: float = 0.12,
        deterministic_confidence_threshold: float = 0.9,
        repeat_threshold: int = 2,
        inline_timeout_ms: int = 160,
        inline_with_timeout_ms: int = 120,
        repair_only_timeout_ms: int = 100,
    ) -> None:
        self.low_confidence_threshold = low_confidence_threshold
        self.close_confidence_gap_threshold = close_confidence_gap_threshold
        self.deterministic_confidence_threshold = deterministic_confidence_threshold
        self.repeat_threshold = repeat_threshold
        self.inline_timeout_ms = inline_timeout_ms
        self.inline_with_timeout_ms = inline_with_timeout_ms
        self.repair_only_timeout_ms = repair_only_timeout_ms

    def decide(
        self,
        *,
        state: ConversationState,
        normalized_user_text: str,
        raw_stt_final_text: str | None = None,
        local_intent_top_n: tuple[IntentCandidate, ...] = (),
        selected_local_intent: Intent = Intent.UNKNOWN,
        local_intent_confidence: float = 0.0,
        local_slots: tuple[SlotValue, ...] = (),
        active_pending_item_context: object | None = None,
        available_options_context: Iterable[str] = (),
        fallback_count: int = 0,
        repeated_prompt_count: int = 0,
        previous_turns_summary: tuple[tuple[str, str], ...] = (),
        handler_resolution_status: str | None = None,
        last_response_key: str | None = None,
        duplicate_transcript: bool = False,
    ) -> GptExecutionDecision:
        text = (normalized_user_text or "").strip().lower()
        raw_text = (raw_stt_final_text or normalized_user_text or "").strip().lower()
        options = tuple(
            opt.strip()
            for opt in available_options_context
            if isinstance(opt, str) and opt.strip()
        )
        top_intents = self._top_intents(
            selected_local_intent=selected_local_intent,
            local_intent_confidence=local_intent_confidence,
            local_intent_top_n=local_intent_top_n,
        )
        allowed_intents = tuple(sorted(get_candidates(state.value)))
        base_bucket = self._bucket_for_state(state)
        exact_option_match = self._exact_option_match(text, options)
        likely_phonetic = self._likely_phonetic_option_match(text or raw_text, options)
        unresolved_handler = (handler_resolution_status or "").strip().lower() in {
            "unresolved",
            "failed",
            "no_match",
        }
        has_item_slot = self._has_slot(local_slots, "ITEM")
        has_side_slot = self._has_slot(local_slots, "SIDE")
        has_modifier_slot = self._has_slot(local_slots, "MODIFIER")
        has_variant_slot = self._has_slot(local_slots, "VARIANT")
        has_quantity_slot = self._has_slot(local_slots, "QUANTITY")
        low_confidence = local_intent_confidence < self.low_confidence_threshold
        close_top2 = self._top2_close(local_intent_top_n)
        multi_item = self._looks_like_multi_item(text, selected_local_intent, has_item_slot)
        correction_or_denial = self._looks_like_correction_or_denial(
            text=text,
            state=state,
            previous_turns_summary=previous_turns_summary,
        )
        wants_checkout = any(phrase in text for phrase in _ORDER_EXIT_PHRASES)
        repeated_fallback = (
            fallback_count >= self.repeat_threshold
            or repeated_prompt_count >= self.repeat_threshold
            or (last_response_key or "") == "intent_not_allowed"
        )
        hard_add_item = (
            selected_local_intent == Intent.ADD_ITEM
            and has_item_slot
            and (has_side_slot or has_modifier_slot or has_variant_slot or has_quantity_slot or self._has_text_hint(text))
        )

        if duplicate_transcript or text in _NO_SPEECH_MARKERS:
            return self._decision(
                mode=GptExecutionMode.NONE,
                bucket=base_bucket,
                reasons=("silence_or_duplicate",),
                allowed_intents=allowed_intents,
                top_intents=top_intents,
                state=state,
                confidence=local_intent_confidence,
                timeout_ms=0,
                include_previous_turns_count=0,
                include_constrained_options=False,
                include_pending_item_context=False,
                include_menu_candidates=False,
            )

        if exact_option_match and local_intent_confidence >= self.deterministic_confidence_threshold:
            return self._decision(
                mode=GptExecutionMode.NONE,
                bucket=base_bucket,
                reasons=("exact_option_match",),
                allowed_intents=allowed_intents,
                top_intents=top_intents,
                state=state,
                confidence=local_intent_confidence,
                timeout_ms=0,
                include_previous_turns_count=0,
                include_constrained_options=True,
                include_pending_item_context=bool(active_pending_item_context),
                include_menu_candidates=False,
            )

        if (
            selected_local_intent in _YES_NO_INTENTS
            and local_intent_confidence >= self.deterministic_confidence_threshold
            and state not in _WAITING_SELECTION_STATES
        ):
            return self._decision(
                mode=GptExecutionMode.NONE,
                bucket=base_bucket,
                reasons=("deterministic_confirmation",),
                allowed_intents=allowed_intents,
                top_intents=top_intents,
                state=state,
                confidence=local_intent_confidence,
                timeout_ms=0,
                include_previous_turns_count=0,
                include_constrained_options=False,
                include_pending_item_context=False,
                include_menu_candidates=False,
            )

        if repeated_fallback:
            return self._decision(
                mode=GptExecutionMode.INLINE_WITH_TIMEOUT,
                bucket=GptPromptBucket.FALLBACK_RECOVERY,
                reasons=("repeated_fallback",),
                allowed_intents=allowed_intents,
                top_intents=top_intents,
                state=state,
                confidence=local_intent_confidence,
                timeout_ms=self.inline_with_timeout_ms,
                include_previous_turns_count=3,
                include_constrained_options=bool(options),
                include_pending_item_context=bool(active_pending_item_context),
                include_menu_candidates=False,
            )

        if correction_or_denial:
            return self._decision(
                mode=GptExecutionMode.INLINE,
                bucket=GptPromptBucket.CORRECTION_OR_DENIAL,
                reasons=("correction_or_denial",),
                allowed_intents=allowed_intents,
                top_intents=top_intents,
                state=state,
                confidence=local_intent_confidence,
                timeout_ms=self.inline_timeout_ms,
                include_previous_turns_count=2,
                include_constrained_options=bool(options),
                include_pending_item_context=True,
                include_menu_candidates=True,
            )

        if multi_item:
            return self._decision(
                mode=GptExecutionMode.INLINE,
                bucket=GptPromptBucket.ADD_ITEM_PLAN,
                reasons=("multi_item_utterance",),
                allowed_intents=allowed_intents,
                top_intents=top_intents,
                state=state,
                confidence=local_intent_confidence,
                timeout_ms=self.inline_timeout_ms,
                include_previous_turns_count=2,
                include_constrained_options=False,
                include_pending_item_context=bool(active_pending_item_context),
                include_menu_candidates=True,
            )

        if hard_add_item:
            return self._decision(
                mode=GptExecutionMode.INLINE,
                bucket=GptPromptBucket.ADD_ITEM_PLAN,
                reasons=("hard_add_item_turn",),
                allowed_intents=allowed_intents,
                top_intents=top_intents,
                state=state,
                confidence=local_intent_confidence,
                timeout_ms=self.inline_timeout_ms,
                include_previous_turns_count=2,
                include_constrained_options=bool(options),
                include_pending_item_context=True,
                include_menu_candidates=True,
            )

        if wants_checkout and state in _WAITING_SELECTION_STATES:
            return self._decision(
                mode=GptExecutionMode.INLINE,
                bucket=GptPromptBucket.CHECKOUT_OR_PAYMENT,
                reasons=("checkout_while_waiting",),
                allowed_intents=allowed_intents,
                top_intents=top_intents,
                state=state,
                confidence=local_intent_confidence,
                timeout_ms=self.inline_timeout_ms,
                include_previous_turns_count=2,
                include_constrained_options=bool(options),
                include_pending_item_context=True,
                include_menu_candidates=False,
            )

        if state in _WAITING_SIDE_STATES and options and not exact_option_match:
            reasons = ["waiting_for_side_unresolved_option"]
            if likely_phonetic:
                reasons.append("phonetic_option_mismatch")
            if wants_checkout:
                reasons.append("exit_phrase_in_waiting_state")
            return self._decision(
                mode=GptExecutionMode.INLINE_WITH_TIMEOUT,
                bucket=GptPromptBucket.SIDE_SELECTION,
                reasons=tuple(reasons),
                allowed_intents=allowed_intents,
                top_intents=top_intents,
                state=state,
                confidence=local_intent_confidence,
                timeout_ms=self.inline_with_timeout_ms,
                include_previous_turns_count=2,
                include_constrained_options=True,
                include_pending_item_context=True,
                include_menu_candidates=False,
            )

        if state in _WAITING_MODIFIER_STATES and options and not exact_option_match:
            reasons = ["waiting_for_modifier_unresolved_option"]
            if likely_phonetic:
                reasons.append("phonetic_option_mismatch")
            if wants_checkout:
                reasons.append("exit_phrase_in_waiting_state")
            return self._decision(
                mode=GptExecutionMode.INLINE_WITH_TIMEOUT,
                bucket=GptPromptBucket.MODIFIER_SELECTION,
                reasons=tuple(reasons),
                allowed_intents=allowed_intents,
                top_intents=top_intents,
                state=state,
                confidence=local_intent_confidence,
                timeout_ms=self.inline_with_timeout_ms,
                include_previous_turns_count=2,
                include_constrained_options=True,
                include_pending_item_context=True,
                include_menu_candidates=False,
            )

        if unresolved_handler and local_slots:
            return self._decision(
                mode=GptExecutionMode.INLINE_WITH_TIMEOUT,
                bucket=base_bucket,
                reasons=("slots_present_handler_unresolved",),
                allowed_intents=allowed_intents,
                top_intents=top_intents,
                state=state,
                confidence=local_intent_confidence,
                timeout_ms=self.inline_with_timeout_ms,
                include_previous_turns_count=2,
                include_constrained_options=bool(options),
                include_pending_item_context=bool(active_pending_item_context),
                include_menu_candidates=has_item_slot,
            )

        if (
            selected_local_intent == Intent.UNKNOWN
            or close_top2
            or low_confidence
            or unresolved_handler
        ):
            reasons: list[str] = []
            if selected_local_intent == Intent.UNKNOWN:
                reasons.append("unknown_intent")
            if close_top2:
                reasons.append("top1_top2_close")
            if low_confidence:
                reasons.append("low_local_confidence")
            if unresolved_handler:
                reasons.append("handler_unresolved")
            bucket = base_bucket
            if state in _WAITING_SIDE_STATES:
                bucket = GptPromptBucket.SIDE_SELECTION
            elif state in _WAITING_MODIFIER_STATES:
                bucket = GptPromptBucket.MODIFIER_SELECTION
            return self._decision(
                mode=GptExecutionMode.INLINE_WITH_TIMEOUT,
                bucket=bucket,
                reasons=tuple(reasons),
                allowed_intents=allowed_intents,
                top_intents=top_intents,
                state=state,
                confidence=local_intent_confidence,
                timeout_ms=self.inline_with_timeout_ms,
                include_previous_turns_count=2,
                include_constrained_options=bool(options),
                include_pending_item_context=bool(active_pending_item_context),
                include_menu_candidates=has_item_slot,
            )

        if selected_local_intent == Intent.ADD_ITEM and local_intent_confidence >= self.deterministic_confidence_threshold:
            return self._decision(
                mode=GptExecutionMode.SHADOW,
                bucket=GptPromptBucket.ADD_ITEM_PLAN,
                reasons=("clean_high_conf_add_item",),
                allowed_intents=allowed_intents,
                top_intents=top_intents,
                state=state,
                confidence=local_intent_confidence,
                timeout_ms=0,
                include_previous_turns_count=1,
                include_constrained_options=False,
                include_pending_item_context=bool(active_pending_item_context),
                include_menu_candidates=True,
            )

        if local_intent_confidence >= self.deterministic_confidence_threshold:
            return self._decision(
                mode=GptExecutionMode.SHADOW,
                bucket=base_bucket,
                reasons=("high_confidence_shadow_sampling",),
                allowed_intents=allowed_intents,
                top_intents=top_intents,
                state=state,
                confidence=local_intent_confidence,
                timeout_ms=0,
                include_previous_turns_count=1,
                include_constrained_options=False,
                include_pending_item_context=bool(active_pending_item_context),
                include_menu_candidates=False,
            )

        return self._decision(
            mode=GptExecutionMode.NONE,
            bucket=base_bucket,
            reasons=("no_policy_trigger",),
            allowed_intents=allowed_intents,
            top_intents=top_intents,
            state=state,
            confidence=local_intent_confidence,
            timeout_ms=0,
            include_previous_turns_count=0,
            include_constrained_options=False,
            include_pending_item_context=False,
            include_menu_candidates=False,
        )

    def _decision(
        self,
        *,
        mode: GptExecutionMode,
        bucket: GptPromptBucket,
        reasons: tuple[str, ...],
        allowed_intents: tuple[str, ...],
        top_intents: tuple[GptTopIntent, ...],
        state: ConversationState,
        confidence: float,
        timeout_ms: int,
        include_previous_turns_count: int,
        include_constrained_options: bool,
        include_pending_item_context: bool,
        include_menu_candidates: bool,
    ) -> GptExecutionDecision:
        return GptExecutionDecision(
            mode=mode,
            prompt_bucket=bucket,
            reason_codes=reasons,
            allowed_intents=allowed_intents,
            top_intents=top_intents,
            state=state.value,
            confidence=confidence,
            timeout_ms=timeout_ms,
            include_previous_turns_count=include_previous_turns_count,
            include_constrained_options=include_constrained_options,
            include_pending_item_context=include_pending_item_context,
            include_menu_candidates=include_menu_candidates,
            should_validate=mode != GptExecutionMode.NONE,
            should_apply_result=mode in {
                GptExecutionMode.INLINE,
                GptExecutionMode.INLINE_WITH_TIMEOUT,
                GptExecutionMode.REPAIR_ONLY,
            },
            should_log_shadow_result=mode == GptExecutionMode.SHADOW,
        )

    @staticmethod
    def _bucket_for_state(state: ConversationState) -> GptPromptBucket:
        if state in _WAITING_SIDE_STATES:
            return GptPromptBucket.SIDE_SELECTION
        if state in _WAITING_MODIFIER_STATES:
            return GptPromptBucket.MODIFIER_SELECTION
        if state == ConversationState.CONFIRMING_ORDER or state == ConversationState.WAITING_FOR_PAYMENT:
            return GptPromptBucket.CHECKOUT_OR_PAYMENT
        if state in {
            ConversationState.MODIFYING_ITEM,
            ConversationState.REMOVING_ITEM,
            ConversationState.CANCELLATION_CONFIRMATION,
        }:
            return GptPromptBucket.CANCEL_OR_REMOVE
        if state in {
            ConversationState.IDLE,
            ConversationState.CONFIRMING_ITEM,
        }:
            return GptPromptBucket.ADD_ITEM_PLAN
        return GptPromptBucket.GENERAL_STATE_REPAIR

    @staticmethod
    def _top_intents(
        *,
        selected_local_intent: Intent,
        local_intent_confidence: float,
        local_intent_top_n: tuple[IntentCandidate, ...],
    ) -> tuple[GptTopIntent, ...]:
        if local_intent_top_n:
            return tuple(
                GptTopIntent(intent=c.canonical_intent, confidence=round(float(c.confidence), 4))
                for c in local_intent_top_n[:4]
            )
        return (
            GptTopIntent(
                intent=selected_local_intent.value,
                confidence=round(float(local_intent_confidence or 0.0), 4),
            ),
        )

    def _top2_close(self, local_intent_top_n: tuple[IntentCandidate, ...]) -> bool:
        if len(local_intent_top_n) < 2:
            return False
        gap = float(local_intent_top_n[0].confidence) - float(local_intent_top_n[1].confidence)
        return gap <= self.close_confidence_gap_threshold

    @staticmethod
    def _has_slot(local_slots: tuple[SlotValue, ...], slot_name: str) -> bool:
        wanted = slot_name.upper()
        return any((getattr(slot, "name", "") or "").upper() == wanted for slot in local_slots)

    @staticmethod
    def _exact_option_match(text: str, options: tuple[str, ...]) -> bool:
        if not text or not options:
            return False
        normalized_options = {opt.lower(): opt for opt in options}
        return text in normalized_options

    @staticmethod
    def _looks_like_multi_item(text: str, selected_local_intent: Intent, has_item_slot: bool) -> bool:
        if not text:
            return False
        if selected_local_intent not in {Intent.ADD_ITEM, Intent.UNKNOWN} and not has_item_slot:
            return False
        return any(sep in text for sep in _MULTI_ITEM_SEPARATORS)

    @staticmethod
    def _has_text_hint(text: str) -> bool:
        return any(hint in text for hint in _SIDE_OR_MODIFIER_HINTS)

    @staticmethod
    def _looks_like_correction_or_denial(
        *,
        text: str,
        state: ConversationState,
        previous_turns_summary: tuple[tuple[str, str], ...],
    ) -> bool:
        if not text:
            return False
        has_denial_prefix = any(text.startswith(prefix) for prefix in _DENIAL_PREFIXES)
        if not has_denial_prefix:
            return False
        if state == ConversationState.CONFIRMING_ITEM:
            return True
        return any(role == "bot" for role, _ in previous_turns_summary)

    @staticmethod
    def _likely_phonetic_option_match(text: str, options: tuple[str, ...]) -> bool:
        if not text or not options:
            return False
        condensed = text.replace(" ", "")
        best = 0.0
        for option in options:
            score = SequenceMatcher(None, condensed, option.lower().replace(" ", "")).ratio()
            if score > best:
                best = score
        return best >= 0.68
