from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher

from app.menu.repository import MenuRepository
from app.nlu.intent_resolution.intent import Intent
from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.session.session import Session
from app.state_machine.control_intent_resolver import (
    ControlIntentKind,
    log_control_intent_event,
    resolve_control_intent,
)
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.pending_item_models import InterruptProposal, PendingVariantChoice
from app.state_machine.models.conversation_state import ConversationState
from app.state_machine.handler_result import HandlerResult
from app.state_machine.handlers.base_handler import BaseHandler
from app.state_machine.handlers.item.add_item.add_item_flow import (
    ReadyToFinalize,
    determine_next_add_item_step,
)
from app.state_machine.handlers.item.add_item.option_matching import (
    build_match_debug_payload,
    extract_slot_candidate_texts,
    score_scoped_choice,
)
from app.utils.candidate_texts import build_candidate_texts_normalized
from app.utils.token_matcher import tokenize

from app.state_machine.flow_sets import SOFT_SWITCH_INTENTS

AUTO_ACCEPT_THRESHOLD = 0.80
CONFIRM_THRESHOLD = 0.60
MIN_CONFIRM_GAP = 0.08


@dataclass(frozen=True, slots=True)
class _ScoredVariantChoice:
    variant_id: str
    choice_name: str
    confidence: float


def _first_size_slot_normalized(slots) -> str | None:
    values = extract_slot_candidate_texts(
        slots=slots,
        allowed_slot_labels={"SIZE", "VARIANT"},
    )
    return values[0] if values else None


def _looks_like_pure_size_answer(
    normalized_user_text: str,
    normalized_choice_names: tuple[str, ...],
) -> bool:
    if not normalized_user_text:
        return False

    filler_words = {
        "please",
        "the",
        "a",
        "an",
        "one",
        "size",
        "with",
        "thanks",
        "thank",
        "you",
        "um",
        "uh",
        "okay",
        "ok",
        "ill",
        "i",
        "want",
        "would",
        "like",
        "to",
        "have",
        "my",
        "make",
        "it",
        "get",
        "take",
    }
    blocked_phrases = {
        "how much",
        "price",
        "cost",
        "add",
        "show",
        "menu",
        "checkout",
        "check out",
        "cart",
        "total",
        "remove",
        "change",
        "modify",
        "instead",
    }

    if any(phrase in normalized_user_text for phrase in blocked_phrases):
        return False

    tokens = [token for token in normalized_user_text.split() if token not in filler_words]
    compact = " ".join(tokens).strip()
    if not compact:
        return False

    if compact in normalized_choice_names:
        return True

    for choice_name in normalized_choice_names:
        if len(compact) >= 3 and (compact in choice_name or choice_name in compact):
            return True

    return False


class _VariantMatchMixin:
    def _similarity_ratio(self, a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

    def _choice_confidence(self, candidate: str, choice_name: str) -> float:
        if not candidate or not choice_name:
            return 0.0
        if candidate == choice_name:
            return 1.0

        best = 0.0
        candidate_tokens = set(tokenize(candidate))
        choice_tokens = set(tokenize(choice_name))

        if candidate_tokens and choice_tokens:
            overlap = len(candidate_tokens & choice_tokens)
            coverage = overlap / len(choice_tokens)
            candidate_coverage = overlap / len(candidate_tokens)
            best = max(best, max(coverage, candidate_coverage))

        return max(
            best,
            score_scoped_choice(
                candidate,
                choice_name,
                reject_candidate_superset=False,
            ),
        )

    def _resolve_best_variant_from_values(
        self,
        *,
        normalized_values: list[str],
        choices_by_normalized_name: dict[str, PendingVariantChoice],
    ) -> _ScoredVariantChoice | None:
        candidate_texts = build_candidate_texts_normalized(
            normalized_user_text="",
            normalized_slot_values=normalized_values,
            allow_split=True,
        )
        if not candidate_texts:
            return None

        best_choice: PendingVariantChoice | None = None
        best_confidence = 0.0
        second_confidence = 0.0

        for choice in choices_by_normalized_name.values():
            choice_best = 0.0
            for candidate in candidate_texts:
                choice_best = max(choice_best, self._choice_confidence(candidate, choice.normalized_name))

            if choice_best > best_confidence:
                second_confidence = best_confidence
                best_confidence = choice_best
                best_choice = choice
            elif choice_best > second_confidence:
                second_confidence = choice_best

        if best_choice is None or best_confidence < CONFIRM_THRESHOLD:
            return None

        if best_confidence < AUTO_ACCEPT_THRESHOLD and (best_confidence - second_confidence) < MIN_CONFIRM_GAP:
            return None

        return _ScoredVariantChoice(
            variant_id=best_choice.variant_id,
            choice_name=best_choice.name,
            confidence=best_confidence,
        )


class WaitingForSideSizeHandler(BaseHandler, _VariantMatchMixin):
    def __init__(self, menu_repo: MenuRepository | None = None) -> None:
        self.menu_repo = menu_repo
        # Bucket 2: WaitingOptionResolver — lazy-initialized on first call.
        self._waiting_resolver: object | None = None

    def handle(
        self,
        intent: Intent,
        context: ConversationContext,
        user_text: str,
        session: Session | None = None,
    ) -> HandlerResult:
        pending = context.pending_add_item
        pending_side_item_id = context.pending_side_item_id

        if pending is None or not pending_side_item_id:
            return HandlerResult(
                next_state=ConversationState.ERROR_RECOVERY,
                response_key="item_context_missing",
            )

        normalized_user_text = user_text or ""
        side_choice = pending.side_choice_by_item_id.get(pending_side_item_id)

        if side_choice is None:
            return HandlerResult(
                next_state=ConversationState.ERROR_RECOVERY,
                response_key="item_context_missing",
            )

        if side_choice.pricing_mode != "variant" or not side_choice.variants:
            self._clear_pending_side_size(context)
            step = determine_next_add_item_step(context)
            return self._step_to_result(context, step)

        available_sizes = list(side_choice.variant_names)
        normalized_choice_names = tuple(side_choice.variants_by_normalized_name.keys())
        control_intent = resolve_control_intent(
            normalized_user_text,
            intent,
            getattr(context.last_nlu, "model_sub_intent", None),
            ConversationState.WAITING_FOR_SIDE_SIZE,
            context,
            nlu_result=context.last_nlu,
            intent_confidence=context.last_intent_confidence,
        )

        pending_confirmation = self._get_pending_side_size_confirmation(
            context=context,
            side_item_id=side_choice.item_id,
        )
        if pending_confirmation is not None:
            if control_intent is not None and control_intent.kind == ControlIntentKind.AFFIRM:
                context.selected_side_variants[side_choice.item_id] = pending_confirmation["variant_id"]
                self._clear_pending_side_size(context)
                self._clear_pending_side_size_confirmation(context, side_choice.item_id)
                step = determine_next_add_item_step(context)
                return self._step_to_result(context, step)

            if control_intent is not None and control_intent.kind == ControlIntentKind.CANCEL:
                log_control_intent_event(
                    "control_intent_action",
                    state=ConversationState.WAITING_FOR_SIDE_SIZE.value,
                    action="cancel_pending_item",
                    kind=control_intent.kind.value,
                )
                context.reset_item_scope()
                return HandlerResult(
                    next_state=ConversationState.IDLE,
                    response_key="item_cancelled_successfully",
                )

            if control_intent is not None and control_intent.kind in {
                ControlIntentKind.DENY,
                ControlIntentKind.META_CLARIFY,
                ControlIntentKind.OPTIONS_REQUEST,
            }:
                if control_intent.kind == ControlIntentKind.META_CLARIFY:
                    log_control_intent_event(
                        "meta_clarify_repeated",
                        state=ConversationState.WAITING_FOR_SIDE_SIZE.value,
                        field_name="side_size",
                    )
                if control_intent.kind == ControlIntentKind.OPTIONS_REQUEST:
                    log_control_intent_event(
                        "options_requested",
                        state=ConversationState.WAITING_FOR_SIDE_SIZE.value,
                        field_name="side_size",
                    )
                self._clear_pending_side_size_confirmation(context, side_choice.item_id)
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_SIDE_SIZE,
                    response_key="repeat_side_size_options",
                    response_payload={
                        "item_name": pending.item_name,
                        "side_item_name": side_choice.name,
                        "available_sizes": available_sizes,
                    },
                )

            self._clear_pending_side_size_confirmation(context, side_choice.item_id)

        if control_intent is not None:
            if control_intent.kind == ControlIntentKind.CANCEL:
                log_control_intent_event(
                    "control_intent_action",
                    state=ConversationState.WAITING_FOR_SIDE_SIZE.value,
                    action="cancel_pending_item",
                    kind=control_intent.kind.value,
                )
                context.reset_item_scope()
                return HandlerResult(
                    next_state=ConversationState.IDLE,
                    response_key="item_cancelled_successfully",
                )

            if control_intent.kind == ControlIntentKind.META_CLARIFY:
                log_control_intent_event(
                    "meta_clarify_repeated",
                    state=ConversationState.WAITING_FOR_SIDE_SIZE.value,
                    field_name="side_size",
                )
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_SIDE_SIZE,
                    response_key="repeat_side_size_options",
                    response_payload={
                        "item_name": pending.item_name,
                        "side_item_name": side_choice.name,
                        "available_sizes": available_sizes,
                    },
                )

            if control_intent.kind == ControlIntentKind.OPTIONS_REQUEST:
                log_control_intent_event(
                    "options_requested",
                    state=ConversationState.WAITING_FOR_SIDE_SIZE.value,
                    field_name="side_size",
                )
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_SIDE_SIZE,
                    response_key="repeat_side_size_options",
                    response_payload={
                        "item_name": pending.item_name,
                        "side_item_name": side_choice.name,
                        "available_sizes": available_sizes,
                    },
                )

            if control_intent.kind in {ControlIntentKind.DENY, ControlIntentKind.DONE}:
                log_control_intent_event(
                    "required_selection_cannot_skip",
                    state=ConversationState.WAITING_FOR_SIDE_SIZE.value,
                    field_name="side_size",
                )
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_SIDE_SIZE,
                    response_key="required_side_size_cannot_skip",
                    response_payload={
                        "item_name": pending.item_name,
                        "side_item_name": side_choice.name,
                        "available_sizes": available_sizes,
                    },
                )

            if control_intent.kind == ControlIntentKind.AFFIRM:
                log_control_intent_event(
                    "control_intent_action",
                    state=ConversationState.WAITING_FOR_SIDE_SIZE.value,
                    action="side_size_requires_explicit_choice",
                    kind=control_intent.kind.value,
                )
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_SIDE_SIZE,
                    response_key="repeat_side_size_options",
                    response_payload={
                        "item_name": pending.item_name,
                        "side_item_name": side_choice.name,
                        "available_sizes": available_sizes,
                    },
                )

        scored_match: _ScoredVariantChoice | None = None

        slot_value = _first_size_slot_normalized(context.last_slots or ())
        match_debug = build_match_debug_payload(
            raw_utterance=normalized_user_text,
            candidates=[],
            selected_candidate=slot_value or normalized_user_text or None,
            matched_option=None,
            match_source=("slot_value" if slot_value else ("raw_utterance" if normalized_user_text else None)),
            match_score=None,
        )
        if slot_value:
            scored_match = self._resolve_best_variant_from_values(
                normalized_values=[slot_value],
                choices_by_normalized_name=side_choice.variants_by_normalized_name,
            )

        if scored_match is None and intent in SOFT_SWITCH_INTENTS:
            context.return_state = ConversationState.WAITING_FOR_SIDE_SIZE
            return HandlerResult(
                next_state=ConversationState.CANCELLATION_CONFIRMATION,
                response_key="confirm_cancel_current_item_for_new_request",
                response_payload={"item_name": pending.item_name},
                awaiting_flow_confirmation=True,
                interrupt_proposal=InterruptProposal(
                    text=normalized_user_text,
                    predicted_main_intent=None,
                    predicted_sub_intent=intent.value,
                ),
            )

        if scored_match is None and _looks_like_pure_size_answer(
            normalized_user_text,
            normalized_choice_names,
        ):
            scored_match = self._resolve_best_variant_from_values(
                normalized_values=[normalized_user_text],
                choices_by_normalized_name=side_choice.variants_by_normalized_name,
            )

        if scored_match is not None:
            match_debug = build_match_debug_payload(
                raw_utterance=normalized_user_text,
                candidates=[],
                selected_candidate=slot_value or normalized_user_text or None,
                matched_option=scored_match.choice_name,
                match_source=("slot_value" if slot_value else "raw_utterance"),
                match_score=scored_match.confidence,
            )
            if scored_match.confidence >= AUTO_ACCEPT_THRESHOLD:
                context.selected_side_variants[side_choice.item_id] = scored_match.variant_id
                self._clear_pending_side_size(context)
                self._clear_pending_side_size_confirmation(context, side_choice.item_id)
                step = determine_next_add_item_step(context)
                return self._step_to_result(context, step)

            if scored_match.confidence >= CONFIRM_THRESHOLD:
                self._set_pending_side_size_confirmation(
                    context=context,
                    side_item_id=side_choice.item_id,
                    variant_id=scored_match.variant_id,
                    choice_name=scored_match.choice_name,
                    confidence=scored_match.confidence,
                )
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_SIDE_SIZE,
                    response_key="confirm_side_size_choice_guess",
                    response_payload={
                        "item_name": pending.item_name,
                        "side_item_name": side_choice.name,
                        "choice_name": scored_match.choice_name,
                        **match_debug,
                    },
                )

        # ── Bucket 2: WaitingOptionResolver (GptSafeClient-based) ────────────
        # Reached when deterministic matching found no confident variant match.
        _b2_side_size = self._try_bucket2_resolver_side_size(
            user_text=normalized_user_text,
            pending=pending,
            side_choice=side_choice,
            context=context,
        )
        if _b2_side_size is not None:
            return _b2_side_size
        # ── END Bucket 2 ──────────────────────────────────────────────────────

        return HandlerResult(
            next_state=ConversationState.WAITING_FOR_SIDE_SIZE,
            response_key="invalid_side_size_option",
            response_payload={
                "item_name": pending.item_name,
                "side_item_name": side_choice.name,
                "available_sizes": available_sizes,
                **match_debug,
            },
        )

    # ------------------------------------------------------------------
    # Bucket 2: WaitingOptionResolver helpers
    # ------------------------------------------------------------------

    def _ensure_waiting_resolver(self) -> object:
        """Lazily initialize the WaitingOptionResolver and return it."""
        if self._waiting_resolver is None:
            from app.nlu.turn_resolver.waiting_option_resolver import WaitingOptionResolver
            self._waiting_resolver = WaitingOptionResolver()
        return self._waiting_resolver

    def _try_bucket2_resolver_side_size(
        self,
        *,
        user_text: str,
        pending: object,
        side_choice: object,
        context: "ConversationContext",
    ) -> "HandlerResult | None":
        """Try bucket-2 WaitingOptionResolver for side-size/variant resolution.

        Returns a HandlerResult when a variant was resolved and applied.
        Returns None to fall through to the invalid_side_size_option fallback.
        Never raises.
        """
        try:
            from app.config.semantic_repair import get_semantic_repair_config
            cfg = get_semantic_repair_config()
            if getattr(cfg, "bucket_2_mode", "disabled") == "disabled":
                return None

            import logging as _logging
            from app.nlu.turn_resolver.waiting_option_policy import (
                should_call_waiting_option_gpt,
            )
            from app.nlu.turn_resolver.waiting_option_validator import (
                validate_waiting_option_resolution,
            )
            from app.nlu.turn_resolver.waiting_option_resolver import WaitingOptionAction
            from app.nlu.turn_resolver.allowed_option_extractor import AllowedOptionExtractor
            from app.nlu.query_normalization.text_preprocessor import normalize_text as _norm

            _log = _logging.getLogger(__name__)

            state = ConversationState.WAITING_FOR_SIDE_SIZE.value
            last_nlu = getattr(context, "last_nlu", None)
            local_intent = str(getattr(last_nlu, "intent", None) or "") if last_nlu else ""
            local_confidence = float(context.last_intent_confidence or 0.0)
            local_slots = list(context.last_slots or ())

            should_call, trigger_reason = should_call_waiting_option_gpt(
                state=state,
                user_text=user_text,
                local_intent=local_intent,
                local_confidence=local_confidence,
                local_slots=local_slots,
            )
            if not should_call:
                return None

            resolver = self._ensure_waiting_resolver()
            resolution = resolver.resolve_sync(  # type: ignore[attr-defined]
                context=context,
                user_text=user_text,
                normalized_text=user_text,
                local_intent=local_intent,
                local_confidence=local_confidence,
                local_candidates=None,
                local_slots=local_slots,
                state=state,
            )

            allowed_options = AllowedOptionExtractor().extract(context, state)
            min_conf = float(getattr(cfg, "bucket_2_min_confidence", 0.70))
            validation = validate_waiting_option_resolution(
                resolution, allowed_options, state, context,
                min_confidence=min_conf,
            )
            if not validation.is_valid:
                return None

            action = resolution.action
            pending_name = getattr(pending, "item_name", "")
            side_name = getattr(side_choice, "name", "")
            available = getattr(side_choice, "variant_names", [])

            if action == WaitingOptionAction.LIST_OPTIONS:
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_SIDE_SIZE,
                    response_key="repeat_side_size_options",
                    response_payload={
                        "item_name": pending_name,
                        "side_item_name": side_name,
                        "available_sizes": list(available),
                        "list_options_requested": True,
                    },
                )

            if action == WaitingOptionAction.CLARIFY:
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_SIDE_SIZE,
                    response_key="repeat_side_size_options",
                    response_payload={
                        "item_name": pending_name,
                        "side_item_name": side_name,
                        "available_sizes": list(available),
                        "repeat_reason": "clarify",
                        **({"clarification_text": resolution.clarification_text}
                           if resolution.clarification_text else {}),
                    },
                )

            # SELECT — find variant by name in side_choice.variants_by_normalized_name
            if action == WaitingOptionAction.SELECT and resolution.ok:
                gpt_name = (
                    resolution.selected_option_names[0]
                    if resolution.selected_option_names
                    else (resolution.selected_variant or "")
                )
                if not gpt_name:
                    return None

                variants_by_norm = getattr(side_choice, "variants_by_normalized_name", {})
                norm_gpt = _norm(gpt_name) if gpt_name else ""
                variant_choice = variants_by_norm.get(norm_gpt)
                if variant_choice is None:
                    gpt_lower = gpt_name.lower()
                    for vc in variants_by_norm.values():
                        if getattr(vc, "name", "").lower() == gpt_lower:
                            variant_choice = vc
                            break

                if variant_choice is None:
                    return None

                side_item_id = getattr(side_choice, "item_id", "")
                variant_id = getattr(variant_choice, "variant_id", "")
                if not side_item_id or not variant_id:
                    return None

                _log.info(
                    "waiting_option_gpt_applied",
                    extra={
                        "event": "waiting_option_gpt_applied",
                        "waiting_option_gpt_applied": True,
                        "final_option_source": "gpt",
                        "variant_name": getattr(variant_choice, "name", ""),
                        "state": state,
                    },
                )
                context.selected_side_variants[side_item_id] = variant_id
                self._clear_pending_side_size(context)
                self._clear_pending_side_size_confirmation(context, side_item_id)
                step = determine_next_add_item_step(context)
                return self._step_to_result(context, step)

            return None

        except Exception as exc:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "waiting_option_bucket2_side_size_error",
                extra={
                    "event": "waiting_option_bucket2_side_size_error",
                    "error": str(exc)[:200],
                    "state": ConversationState.WAITING_FOR_SIDE_SIZE.value,
                },
            )
            return None

    def _set_pending_side_size_confirmation(
        self,
        *,
        context: ConversationContext,
        side_item_id: str,
        variant_id: str,
        choice_name: str,
        confidence: float,
    ) -> None:
        context.awaiting_confirmation_for = {
            "type": "side_size_choice_guess",
            "side_item_id": side_item_id,
            "variant_id": variant_id,
            "choice_name": choice_name,
            "confidence": confidence,
        }

    def _get_pending_side_size_confirmation(
        self,
        *,
        context: ConversationContext,
        side_item_id: str,
    ) -> dict | None:
        confirmation = getattr(context, "awaiting_confirmation_for", None)
        if not isinstance(confirmation, dict):
            return None
        if confirmation.get("type") != "side_size_choice_guess":
            return None
        if confirmation.get("side_item_id") != side_item_id:
            return None
        return confirmation

    def _clear_pending_side_size_confirmation(
        self,
        context: ConversationContext,
        side_item_id: str,
    ) -> None:
        confirmation = getattr(context, "awaiting_confirmation_for", None)
        if (
            isinstance(confirmation, dict)
            and confirmation.get("type") == "side_size_choice_guess"
            and confirmation.get("side_item_id") == side_item_id
        ):
            context.awaiting_confirmation_for = None

    def _clear_pending_side_size(self, context: ConversationContext) -> None:
        context.pending_side_item_id = None
        context.pending_side_item_name = None
        context.pending_side_group_id = None

        if context.current_prompt_field == "side_size":
            context.current_prompt_field = None

        if context.available_choices_kind == "side_size":
            context.available_choices_kind = None
            context.available_choices_values = ()

    def _step_to_result(self, context: ConversationContext, step) -> HandlerResult:
        pending = context.pending_add_item
        if pending is None:
            return HandlerResult(
                next_state=ConversationState.ERROR_RECOVERY,
                response_key="item_context_missing",
            )

        if isinstance(step, ReadyToFinalize):
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="item_added_successfully",
                response_payload={
                    "item_name": pending.item_name,
                    "quantity": context.quantity or 1,
                },
                command=step.command.to_dict(),
                reset_context=True,
            )

        return HandlerResult(
            next_state=step.next_state,
            response_key=step.response_key,
            response_payload=step.response_payload,
        )
