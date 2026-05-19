# app/state_machine/handlers/item/add_item/waiting_for_modifier_handler.py
from __future__ import annotations

import logging
from dataclasses import dataclass

_logger = logging.getLogger(__name__)

from app.menu.repository import MenuRepository
from app.nlu.intent_resolution.intent import Intent
from app.session.session import Session
from app.state_machine.control_intent_resolver import (
    ControlIntentKind,
    log_control_intent_event,
    resolve_control_intent,
)
from app.state_machine.handlers.item.add_item.group_classification import speech_noun_for_modifier_group
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.pending_item_models import InterruptProposal, PendingModifierGroup
from app.state_machine.models.conversation_state import ConversationState
from app.state_machine.models.pending_item_models import ModifierSelection
from app.state_machine.handler_result import HandlerResult
from app.state_machine.handlers.item.add_item.group_resolution_handler import GroupResolutionHandler
from app.state_machine.handlers.item.add_item.add_item_handler import PendingItemCaptureHelper
from app.state_machine.handlers.item.add_item.add_item_flow import (
    ReadyToFinalize,
    determine_next_add_item_step,
)
from app.state_machine.handlers.item.add_item.modifier_group_resolver import (
    ModifierGroupResolver,
    build_modifier_option_candidates,
    extract_modifier_slot_values_normalized,
)
from app.state_machine.handlers.item.add_item.group_collection_utils import (
    effective_group_selector_bounds,
)
from app.state_machine.handlers.item.add_item.group_skip_policy import (
    GroupSkipDecision,
    evaluate_group_skip,
)

from app.state_machine.flow_sets import (
    SOFT_SWITCH_INTENTS_REDUCED as SOFT_SWITCH_INTENTS,
    GROUP_DONE_INTENTS,
    looks_like_done_answer,
    looks_like_more_options_answer,
    looks_like_skip_answer,
)
from app.nlu.control_phrase_classifier import DEFAULT_CLASSIFIER
from app.nlu.utterance_filter import DEFAULT_FILTER
from app.nlu.query_normalization.text_preprocessor import normalize_text as _normalize_text
from app.state_machine.handlers.item.add_item.waiting_state_interruption_policy import (
    InterruptionDecision,
    evaluate_waiting_modifier_interruption,
)
from app.nlu.semantic_repair.option_resolver_result import (
    OPTION_RESOLVER_NOT_CALLED,
    OptionResolverResult,
)
from app.nlu.semantic_repair.option_selection_validator import (
    build_modifier_selections_from_names,
)


def _looks_like_done_answer(normalized_user_text: str) -> bool:
    return looks_like_done_answer(normalized_user_text)


def _looks_like_more_options(normalized_user_text: str) -> bool:
    return looks_like_more_options_answer(normalized_user_text)


def _looks_like_skip_modifier_answer(normalized_user_text: str, group: PendingModifierGroup) -> bool:
    text = (normalized_user_text or "").strip()
    if not text:
        return False

    # whole-group skip only; specific "no onions" is handled by the resolver
    return looks_like_skip_answer(text)


def _looks_like_specific_modifier_removal(normalized_user_text: str) -> bool:
    text = (normalized_user_text or "").strip()
    return bool(text) and (text.startswith("no ") or text.startswith("without "))


@dataclass(frozen=True, slots=True)
class _PrefilledModifierGroups:
    matched_names: list[str]
    applied: bool
    overflow_group_id: str | None = None
    overflow_requested_names: list[str] | None = None
    overflow_unmatched_names: list[str] | None = None
    overflow_max_selector: int = 0


class WaitingForModifierHandler(GroupResolutionHandler):
    """
    Resolve modifier selections strictly from the active pending item snapshot.

    Important:
    - no broad menu resolution during waiting state
    - only choices from the current active modifier group can match
    - supports multi-modifier capture in one utterance
    - supports structured selections like:
        - bacon
        - extra bacon
        - no onions
        - less mayo
    - keeps group open when min is met but more are still allowed
    """

    def __init__(self, menu_repo: MenuRepository | None = None) -> None:
        self.menu_repo = menu_repo
        self.modifier_resolver = ModifierGroupResolver()
        self.capture_helper = PendingItemCaptureHelper(
            modifier_resolver=self.modifier_resolver,
        )
        # Phase 3: GPT Option Resolver — lazy-initialized on first call.
        # This is None until _ensure_option_resolver() is invoked.
        self._option_resolver: object | None = None
        # Bucket 2: WaitingOptionResolver (GptSafeClient-based) — lazy-initialized.
        # This is None until _ensure_waiting_resolver() is invoked.
        self._waiting_resolver: object | None = None

    def handle(
        self,
        intent: Intent,
        context: ConversationContext,
        user_text: str,
        session: Session | None = None,
    ) -> HandlerResult:
        pending = context.pending_add_item
        if pending is None:
            return HandlerResult(
                next_state=ConversationState.ERROR_RECOVERY,
                response_key="item_context_missing",
            )

        normalized_user_text = user_text or ""
        groups = pending.modifier_groups
        idx = context.current_modifier_group_index

        if idx >= len(groups):
            step = determine_next_add_item_step(context)
            return self._step_to_result(context, step)

        group = groups[idx]
        existing_selections = list(context.selected_modifier_groups.get(group.group_id, []))
        existing_ids = [sel.modifier_id for sel in existing_selections]
        self.capture_helper.prefill_quantity(
            context=context,
            user_text=normalized_user_text,
        )

        min_selector, max_selector = effective_group_selector_bounds(group)

        # ── Control-phrase pre-classification ────────────────────────────
        # Intercept skip/done/repeat BEFORE the resolver so phrases like
        # "no skip that", "can you repeat", "add done" never reach
        # unmatched_names and get echoed back.
        # For negated_option: if the target is NOT in the modifier group
        # we give a neutral clarification instead of "not available".
        # If the target IS in the group we pass through to the resolver
        # which handles "no onions", "without mayo", etc. correctly.
        _cp = DEFAULT_CLASSIFIER.classify(
            normalized_user_text, ConversationState.WAITING_FOR_MODIFIER.value
        )
        if _cp.action == "repeat":
            log_control_intent_event(
                "control_phrase_classifier_repeat",
                state=ConversationState.WAITING_FOR_MODIFIER.value,
                group_id=group.group_id,
            )
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_MODIFIER,
                response_key="repeat_modifier_options",
                response_payload={
                    **self._choice_payload(group, existing_selections),
                    "repeat_reason": "meta_clarify",
                },
            )

        if _cp.action in {"skip", "done"}:
            log_control_intent_event(
                "control_phrase_classifier_skip_done",
                state=ConversationState.WAITING_FOR_MODIFIER.value,
                action=_cp.action,
                group_id=group.group_id,
            )
            _skip = evaluate_group_skip(min_selector, len(existing_selections))
            if _skip.decision == GroupSkipDecision.BLOCK_UNDER_MIN:
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_MODIFIER,
                    response_key="required_modifier_cannot_skip",
                    response_payload={
                        **self._choice_payload(group, existing_selections),
                        "remaining_to_min": _skip.remaining_to_min,
                        "selected_count": _skip.selected_count,
                        "min_required": _skip.min_required,
                        "intent_kind": _cp.action,
                    },
                )
            if _skip.decision == GroupSkipDecision.SKIP_OPTIONAL and not existing_selections:
                context.skipped_modifier_groups.add(group.group_id)
                context.selected_modifier_groups.pop(group.group_id, None)
            elif _skip.decision == GroupSkipDecision.ADVANCE_MIN_MET:
                log_control_intent_event(
                    "advance_min_met",
                    state=ConversationState.WAITING_FOR_MODIFIER.value,
                    field_name="modifier",
                    group_id=group.group_id,
                    selected_count=_skip.selected_count,
                    min_required=_skip.min_required,
                    kind=_cp.action,
                )
            step = determine_next_add_item_step(context)
            return self._step_to_result(context, step)

        if _cp.action == "negated_option":
            target = _cp.normalized_target or ""
            # If the target matches a choice in this group, pass through to
            # the resolver which handles "no onions", "without mayo" etc.
            if target and self._target_in_modifier_group(target, group):
                pass  # fall through to resolve_control_intent + resolver
            else:
                # Target not in this group — give neutral clarification;
                # do NOT say "unavailable" for what is a control phrase.
                log_control_intent_event(
                    "control_phrase_classifier_negated_not_in_group",
                    state=ConversationState.WAITING_FOR_MODIFIER.value,
                    target=target,
                    group_id=group.group_id,
                )
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_MODIFIER,
                    response_key="repeat_modifier_options",
                    response_payload={
                        **self._choice_payload(group, existing_selections),
                        "repeat_reason": "need_choice",
                    },
                )
        # ── END control-phrase pre-classification ─────────────────────────

        control_intent = resolve_control_intent(
            normalized_user_text,
            intent,
            getattr(context.last_nlu, "model_sub_intent", None),
            ConversationState.WAITING_FOR_MODIFIER,
            context,
            nlu_result=context.last_nlu,
            intent_confidence=context.last_intent_confidence,
        )

        if control_intent is not None:
            if control_intent.kind == ControlIntentKind.OPTIONS_REQUEST:
                log_control_intent_event(
                    "options_requested",
                    state=ConversationState.WAITING_FOR_MODIFIER.value,
                    field_name="modifier",
                    group_id=group.group_id,
                )
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_MODIFIER,
                    response_key="list_modifier_options",
                    response_payload=self._choice_payload(group, existing_selections),
                )

            if control_intent.kind == ControlIntentKind.META_CLARIFY:
                log_control_intent_event(
                    "meta_clarify_repeated",
                    state=ConversationState.WAITING_FOR_MODIFIER.value,
                    field_name="modifier",
                    group_id=group.group_id,
                )
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_MODIFIER,
                    response_key="repeat_modifier_options",
                    response_payload={
                        **self._choice_payload(group, existing_selections),
                        "repeat_reason": "meta_clarify",
                    },
                )

            if control_intent.kind == ControlIntentKind.CANCEL:
                log_control_intent_event(
                    "control_intent_action",
                    state=ConversationState.WAITING_FOR_MODIFIER.value,
                    action="cancel_pending_item",
                    kind=control_intent.kind.value,
                )
                context.reset_item_scope()
                return HandlerResult(
                    next_state=ConversationState.IDLE,
                    response_key="item_cancelled_successfully",
                )

            if control_intent.kind == ControlIntentKind.AFFIRM:
                if len(existing_selections) >= min_selector and existing_selections:
                    log_control_intent_event(
                        "control_intent_action",
                        state=ConversationState.WAITING_FOR_MODIFIER.value,
                        action="accept_current_modifier_selection",
                        kind=control_intent.kind.value,
                    )
                    step = determine_next_add_item_step(context)
                    return self._step_to_result(context, step)

                log_control_intent_event(
                    "control_intent_action",
                    state=ConversationState.WAITING_FOR_MODIFIER.value,
                    action="modifier_selection_requires_explicit_choice",
                    kind=control_intent.kind.value,
                )
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_MODIFIER,
                    response_key="repeat_modifier_options",
                    response_payload={
                        **self._choice_payload(group, existing_selections),
                        "repeat_reason": "need_choice",
                    },
                )

            if control_intent.kind in {ControlIntentKind.DENY, ControlIntentKind.DONE}:
                skip = evaluate_group_skip(min_selector, len(existing_selections))

                if skip.decision == GroupSkipDecision.BLOCK_UNDER_MIN:
                    log_control_intent_event(
                        "required_selection_cannot_skip",
                        state=ConversationState.WAITING_FOR_MODIFIER.value,
                        field_name="modifier",
                        group_id=group.group_id,
                    )
                    return HandlerResult(
                        next_state=ConversationState.WAITING_FOR_MODIFIER,
                        response_key="required_modifier_cannot_skip",
                        response_payload={
                            **self._choice_payload(group, existing_selections),
                            "remaining_to_min": skip.remaining_to_min,
                            "selected_count": skip.selected_count,
                            "min_required": skip.min_required,
                            "intent_kind": control_intent.kind.value,
                            "speech_noun": speech_noun_for_modifier_group(
                                group.name, getattr(group, "prompt_noun", None)
                            ),
                        },
                    )

                if skip.decision == GroupSkipDecision.SKIP_OPTIONAL:
                    # Preserve byte-identical legacy behavior: only mark
                    # the group as "skipped" when the user actually had
                    # no selections. When selections exist (min == 0
                    # with prior picks) we just advance with picks intact.
                    if not existing_selections:
                        context.skipped_modifier_groups.add(group.group_id)
                        context.selected_modifier_groups.pop(group.group_id, None)
                        log_control_intent_event(
                            "skipped_optional_group",
                            state=ConversationState.WAITING_FOR_MODIFIER.value,
                            field_name="modifier",
                            group_id=group.group_id,
                        )
                else:
                    # ADVANCE_MIN_MET: selections meet/exceed min — keep
                    # them, do not flag as skipped.
                    log_control_intent_event(
                        "advance_min_met",
                        state=ConversationState.WAITING_FOR_MODIFIER.value,
                        field_name="modifier",
                        group_id=group.group_id,
                        selected_count=skip.selected_count,
                        min_required=skip.min_required,
                        kind=control_intent.kind.value,
                    )

                step = determine_next_add_item_step(context)
                return self._step_to_result(context, step)

        resolution = self.modifier_resolver.resolve(
            group=group,
            normalized_user_text=normalized_user_text,
            option_candidates=build_modifier_option_candidates(context, normalized_user_text),
            normalized_slot_values=extract_modifier_slot_values_normalized(context),
            already_selected_ids=existing_ids,
            known_choice_phrases=self._all_modifier_choice_phrases(pending),
        )

        if resolution.duplicate_names and not resolution.selections:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_MODIFIER,
                response_key="repeat_modifier_options",
                response_payload={
                    **self._choice_payload(group, existing_selections),
                    "repeat_reason": "duplicate",
                    "duplicate_names": resolution.duplicate_names,
                    "speech_noun": speech_noun_for_modifier_group(
                        group.name, getattr(group, "prompt_noun", None)
                    ),
                },
            )

        if resolution.selections:
            return self._apply_modifier_selection(
                context=context,
                pending=pending,
                group=group,
                matched_selections=resolution.selections,
                unmatched_values=resolution.unmatched_values,
                normalized_user_text=normalized_user_text,
                match_debug=resolution.match_debug,
            )

        carried = self._prefill_following_modifier_groups(
            context=context,
            pending=pending,
            start_index=idx + 1,
            normalized_user_text=normalized_user_text,
            consumed_values=self._selected_choice_match_values(group, existing_selections),
        )
        if carried.overflow_group_id and len(existing_selections) >= min_selector:
            overflow_group = pending.modifier_groups_by_id.get(carried.overflow_group_id)
            if overflow_group is not None:
                context.current_modifier_group_index = pending.modifier_groups.index(overflow_group)
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_MODIFIER,
                    response_key="too_many_modifier_choices",
                    response_payload={
                        **self._choice_payload(
                            overflow_group,
                            context.selected_modifier_groups.get(overflow_group.group_id, []),
                        ),
                        "requested_names": carried.overflow_requested_names or [],
                        "dropped_names": carried.overflow_requested_names or [],
                        "unmatched_names": carried.overflow_unmatched_names or [],
                        "max_selector": carried.overflow_max_selector,
                        "over_max": True,
                    },
                )
        if carried.applied:
            if len(existing_selections) < min_selector:
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_MODIFIER,
                    response_key="ask_for_modifier",
                    response_payload={
                        **self._choice_payload(group, existing_selections),
                        "matched_names": carried.matched_names,
                    },
                )

            if not existing_selections:
                context.skipped_modifier_groups.add(group.group_id)
                context.selected_modifier_groups.pop(group.group_id, None)

            step = determine_next_add_item_step(context)
            return self._step_to_result(
                context,
                step,
                matched_names=carried.matched_names,
            )

        # ── Waiting-state interruption policy ─────────────────────────────────
        # Only reached when the resolver AND carry-prefill both found nothing.
        # Detect new-item utterances ("I want X", "could I get a X", …) that
        # should be blocked rather than echoed back as invalid modifier answers
        # when the current group is required and the min is not yet met.
        _intr = evaluate_waiting_modifier_interruption(
            normalized_user_text=_normalize_text(normalized_user_text),
            pending_item_name=pending.item_name,
            group=group,
            selected_count=len(existing_selections),
        )
        if _intr.decision == InterruptionDecision.BLOCK:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_MODIFIER,
                response_key="block_new_item_until_required_done",
                response_payload={
                    **self._choice_payload(group, existing_selections),
                    "pending_item_name": _intr.pending_item_name,
                    "group_prompt_noun": _intr.group_prompt_noun,
                    "remaining_to_min": _intr.remaining_to_min,
                },
            )
        # ── END interruption policy ────────────────────────────────────────────

        # ── SmartTurnPlanner hook (correction / low-confidence modifier) ─────
        # Invoked for self-correction phrases and low-confidence modifier turns
        # BEFORE Phase 3 so that "no I said cheddar" bypasses the option
        # resolver's phonetic-match path and lands directly at the correction
        # handler.  If the planner resolves, we return immediately.
        # If disabled / timed-out / validation fails, falls through to Phase 3.
        _smart_mod = self._try_smart_planner_modifier(
            user_text=normalized_user_text,
            group=group,
            pending=pending,
            existing_selections=existing_selections,
            context=context,
            session=session,
        )
        if _smart_mod is not None:
            return _smart_mod
        # ── END SmartTurnPlanner hook ─────────────────────────────────────────

        # ── Bucket 2: WaitingOptionResolver (GptSafeClient-based) ────────────
        # Runs BEFORE Phase 3. When bucket_2_mode='inline' and the resolution
        # passes validation, the selection is applied and Phase 3 is skipped.
        # On failure, disabled mode, or invalid result → falls through to Phase 3.
        _b2_modifier = self._try_bucket2_resolver(
            user_text=normalized_user_text,
            group=group,
            pending=pending,
            existing_selections=existing_selections,
            context=context,
            deterministic_result=resolution,
        )
        if _b2_modifier is not None:
            return _b2_modifier
        # ── END Bucket 2 ──────────────────────────────────────────────────────

        # ── Phase 3: GPT Option Resolver ─────────────────────────────────────
        # Attempt GPT resolution when local deterministic matching failed.
        # In "shadow" mode the result is logged only (safe_to_apply=False).
        # In "inline" mode the result is applied when the validator approves.
        # This runs before the repeat_modifier_options fallback so GPT can
        # recover misspellings, phonetic variants, and paraphrases.
        if not resolution.selections:
            _gpt_opt = self._try_gpt_option_resolve(
                user_text=normalized_user_text,
                item_name=pending.item_name,
                group=group,
                existing_selections=existing_selections,
                local_resolved=False,
                session=session,
            )
            if (
                _gpt_opt.safe_to_apply
                and _gpt_opt.decision == "select_option"
                and _gpt_opt.selected_names
            ):
                _gpt_selections = build_modifier_selections_from_names(
                    selected_names=_gpt_opt.selected_names,
                    group=group,
                    existing_ids={sel.modifier_id for sel in existing_selections},
                )
                if _gpt_selections:
                    return self._apply_modifier_selection(
                        context=context,
                        pending=pending,
                        group=group,
                        matched_selections=_gpt_selections,
                        normalized_user_text=normalized_user_text,
                    )
        # ── END Phase 3 ───────────────────────────────────────────────────────

        if resolution.unmatched_values:
            _sanitized_unmatched = DEFAULT_FILTER.strip_unmatched(
                v for v in resolution.unmatched_values if v
            )
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_MODIFIER,
                response_key="repeat_modifier_options",
                response_payload={
                    **self._choice_payload(group, existing_selections),
                    "repeat_reason": "invalid",
                    **({"unmatched_names": _sanitized_unmatched} if _sanitized_unmatched else {}),
                    **self._match_debug_payload(resolution.match_debug),
                },
            )

        if intent in SOFT_SWITCH_INTENTS:
            context.return_state = ConversationState.WAITING_FOR_MODIFIER
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

        return HandlerResult(
            next_state=ConversationState.WAITING_FOR_MODIFIER,
            response_key="repeat_modifier_options",
            response_payload={
                **self._choice_payload(group, existing_selections),
                "repeat_reason": "invalid",
                **self._match_debug_payload(resolution.match_debug),
            },
        )

    def _apply_modifier_selection(
        self,
        *,
        context: ConversationContext,
        pending,
        group: PendingModifierGroup,
        matched_selections: list[ModifierSelection],
        unmatched_values: list[str] | None = None,
        normalized_user_text: str,
        match_debug: dict[str, object] | None = None,
    ) -> HandlerResult:
        existing = list(context.selected_modifier_groups.get(group.group_id, []))
        proposed = list(existing)

        existing_ids = {sel.modifier_id for sel in existing}
        for selection in matched_selections:
            if selection.modifier_id not in existing_ids:
                proposed.append(selection)
                existing_ids.add(selection.modifier_id)

        min_selector, max_selector = effective_group_selector_bounds(group)

        # Build feedback lists
        _unmatched = [v for v in (unmatched_values or []) if v]
        newly_added = [sel for sel in proposed if sel.modifier_id not in {s.modifier_id for s in existing}]
        newly_added_names = [sel.name for sel in newly_added]

        # ── Over-max: accept up to limit, tell user what was capped ──
        if max_selector > 0 and len(proposed) > max_selector:
            payload = self._choice_payload(group, existing)
            payload["requested_names"] = newly_added_names
            payload["dropped_names"] = newly_added_names
            payload["unmatched_names"] = _unmatched
            payload["over_max"] = True
            payload.update(self._match_debug_payload(match_debug))
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_MODIFIER,
                response_key="too_many_modifier_choices",
                response_payload=payload,
            )

        context.selected_modifier_groups[group.group_id] = proposed
        context.skipped_modifier_groups.discard(group.group_id)
        self.capture_helper.prefill_quantity(
            context=context,
            user_text=normalized_user_text,
        )

        if len(proposed) < min_selector:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_MODIFIER,
                response_key="ask_for_modifier",
                response_payload={
                    **self._choice_payload(group, proposed),
                    "matched_names": newly_added_names,
                    "unmatched_names": _unmatched,
                    **self._match_debug_payload(match_debug),
                },
            )

        carried = self._prefill_following_modifier_groups(
            context=context,
            pending=pending,
            start_index=context.current_modifier_group_index + 1,
            normalized_user_text=normalized_user_text,
            consumed_values=self._selected_choice_match_values(group, proposed),
        )
        all_matched_names = newly_added_names + carried.matched_names
        if carried.overflow_group_id:
            overflow_group = pending.modifier_groups_by_id.get(carried.overflow_group_id)
            if overflow_group is not None:
                context.current_modifier_group_index = pending.modifier_groups.index(overflow_group)
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_MODIFIER,
                    response_key="too_many_modifier_choices",
                    response_payload={
                        **self._choice_payload(
                            overflow_group,
                            context.selected_modifier_groups.get(overflow_group.group_id, []),
                        ),
                        "matched_names": newly_added_names,
                        "requested_names": carried.overflow_requested_names or [],
                        "dropped_names": carried.overflow_requested_names or [],
                        "unmatched_names": _unmatched + (carried.overflow_unmatched_names or []),
                        "max_selector": carried.overflow_max_selector,
                        "over_max": True,
                        **self._match_debug_payload(match_debug),
                    },
                )
        if carried.applied:
            step = determine_next_add_item_step(context)
            return self._step_to_result(
                context,
                step,
                matched_names=all_matched_names,
                unmatched_names=_unmatched,
                match_debug=match_debug,
            )

        step = determine_next_add_item_step(context)
        return self._step_to_result(
            context, step,
            matched_names=all_matched_names,
            unmatched_names=_unmatched,
            match_debug=match_debug,
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

    def _try_bucket2_resolver(
        self,
        *,
        user_text: str,
        group: "PendingModifierGroup",
        pending: object,
        existing_selections: "list[ModifierSelection]",
        context: "ConversationContext",
        deterministic_result: object = None,
    ) -> "HandlerResult | None":
        """Try bucket-2 WaitingOptionResolver for modifier option resolution.

        Returns a HandlerResult when the resolution was applied or is a
        structural control action (list_options / clarify / skip).
        Returns None to fall through to Phase 3 / unmatched fallback.
        Never raises.
        """
        try:
            from app.config.semantic_repair import get_semantic_repair_config
            cfg = get_semantic_repair_config()
            if getattr(cfg, "bucket_2_mode", "disabled") == "disabled":
                return None

            from app.nlu.turn_resolver.waiting_option_policy import (
                should_call_waiting_option_gpt,
            )
            from app.nlu.turn_resolver.waiting_option_validator import (
                validate_waiting_option_resolution,
            )
            from app.nlu.turn_resolver.waiting_option_resolver import WaitingOptionAction
            from app.nlu.turn_resolver.allowed_option_extractor import AllowedOptionExtractor

            state = ConversationState.WAITING_FOR_MODIFIER.value
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
                deterministic_match_result=deterministic_result,
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
                deterministic_match_result=deterministic_result,
            )

            _logger.info(
                "waiting_option_gpt_trigger",
                extra={
                    "event": "waiting_option_gpt_trigger",
                    "waiting_option_gpt_trigger_reason": trigger_reason,
                    "waiting_option_gpt_action": resolution.action,
                    "waiting_option_gpt_confidence": resolution.confidence,
                    "waiting_option_gpt_status": resolution.raw_gpt_status,
                    "state": state,
                    "group_id": group.group_id,
                },
            )

            # Build allowed_options for validation
            allowed_options = AllowedOptionExtractor().extract(context, state)
            min_conf = float(getattr(cfg, "bucket_2_min_confidence", 0.70))
            validation = validate_waiting_option_resolution(
                resolution, allowed_options, state, context,
                min_confidence=min_conf,
            )
            _logger.info(
                "waiting_option_gpt_validation_result",
                extra={
                    "event": "waiting_option_gpt_validation_result",
                    "waiting_option_gpt_validation_result": (
                        "ok" if validation.is_valid else f"failed:{validation.reason}"
                    ),
                    "state": state,
                    "group_id": group.group_id,
                },
            )

            if not validation.is_valid:
                return None

            action = resolution.action

            if action == WaitingOptionAction.LIST_OPTIONS:
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_MODIFIER,
                    response_key="repeat_modifier_options",
                    response_payload={
                        **self._choice_payload(group, existing_selections),
                        "list_options_requested": True,
                    },
                )

            if action == WaitingOptionAction.CLARIFY:
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_MODIFIER,
                    response_key="repeat_modifier_options",
                    response_payload={
                        **self._choice_payload(group, existing_selections),
                        "repeat_reason": "clarify",
                        **({"clarification_text": resolution.clarification_text}
                           if resolution.clarification_text else {}),
                    },
                )

            if action == WaitingOptionAction.SKIP:
                _b2_min_sel, _ = effective_group_selector_bounds(group)
                _skip = evaluate_group_skip(_b2_min_sel, len(existing_selections))
                if _skip.decision == GroupSkipDecision.BLOCK_UNDER_MIN:
                    return HandlerResult(
                        next_state=ConversationState.WAITING_FOR_MODIFIER,
                        response_key="required_modifier_cannot_skip",
                        response_payload={
                            **self._choice_payload(group, existing_selections),
                            "remaining_to_min": _skip.remaining_to_min,
                            "selected_count": _skip.selected_count,
                            "min_required": _skip.min_required,
                        },
                    )
                if _skip.decision == GroupSkipDecision.SKIP_OPTIONAL and not existing_selections:
                    context.skipped_modifier_groups.add(group.group_id)
                    context.selected_modifier_groups.pop(group.group_id, None)
                step = determine_next_add_item_step(context)
                return self._step_to_result(context, step)

            # Control actions (cancel / checkout / change type) — do not apply;
            # let existing control intent flow handle them.
            if action in {
                WaitingOptionAction.CANCEL,
                WaitingOptionAction.CHECKOUT_REQUEST,
                WaitingOptionAction.CHANGE_ORDER_TYPE,
            }:
                return None

            # SELECT — build modifier selections from GPT-returned names
            if action == WaitingOptionAction.SELECT and resolution.ok:
                selected_names = list(resolution.selected_option_names)
                if not selected_names and resolution.selected_option_ids:
                    # Resolve IDs → names via allowed_options
                    id_to_name = {
                        str(opt.get("modifier_id") or ""): str(opt.get("name") or "")
                        for opt in allowed_options
                    }
                    selected_names = [
                        id_to_name[oid]
                        for oid in resolution.selected_option_ids
                        if oid in id_to_name and id_to_name[oid]
                    ]

                if selected_names:
                    gpt_selections = build_modifier_selections_from_names(
                        selected_names=tuple(selected_names),
                        group=group,
                        existing_ids={sel.modifier_id for sel in existing_selections},
                    )
                    if gpt_selections:
                        _logger.info(
                            "waiting_option_gpt_applied",
                            extra={
                                "event": "waiting_option_gpt_applied",
                                "waiting_option_gpt_applied": True,
                                "final_option_source": "gpt",
                                "selected_names": [s.name for s in gpt_selections],
                                "group_id": group.group_id,
                                "state": state,
                            },
                        )
                        return self._apply_modifier_selection(
                            context=context,
                            pending=pending,
                            group=group,
                            matched_selections=gpt_selections,
                            normalized_user_text=user_text,
                        )

            return None

        except Exception as exc:
            _logger.warning(
                "waiting_option_bucket2_error",
                extra={
                    "event": "waiting_option_bucket2_error",
                    "error": str(exc)[:200],
                    "state": ConversationState.WAITING_FOR_MODIFIER.value,
                },
            )
            return None

    # ------------------------------------------------------------------
    # Phase 3: GPT Option Resolver helpers
    # ------------------------------------------------------------------

    def _ensure_option_resolver(self) -> object:
        """Lazily initialize the GptOptionResolverService and return it."""
        if self._option_resolver is None:
            from app.nlu.semantic_repair.option_resolver_service import (
                GptOptionResolverService,
            )
            self._option_resolver = GptOptionResolverService()
        return self._option_resolver

    def _try_gpt_option_resolve(
        self,
        *,
        user_text: str,
        item_name: str,
        group: PendingModifierGroup,
        existing_selections: list[ModifierSelection],
        local_resolved: bool,
        session: "Session | None",
        last_response_key: str | None = None,
    ) -> "OptionResolverResult":
        """Attempt GPT option resolution. Never raises — returns sentinel on error.

        Parameters
        ----------
        user_text:
            Normalized customer utterance for this turn.
        item_name:
            Name of the item currently being assembled (e.g. "Cheeseburger").
        group:
            The PendingModifierGroup whose choices are the allowed options.
        existing_selections:
            Modifier selections already applied to this group.
        local_resolved:
            True when ModifierGroupResolver already found at least one selection.
        session:
            Current Session (may be None in tests without session injection).
        last_response_key:
            Last bot response key for context (e.g. "ask_for_modifier").
        """
        try:
            from app.nlu.semantic_repair.option_routing_policy import has_correction_signal

            service = self._ensure_option_resolver()

            # Reprompt count for "modifier" field — used for repeat-loop detection.
            repeat_count = 0
            if session is not None:
                counts = getattr(session, "reprompt_count_by_field", {}) or {}
                repeat_count = int(counts.get("modifier", 0) or 0)

            # Detect self-correction phrases ("actually X", "I mean X", …).
            _has_correction = has_correction_signal(user_text)

            # Recent bot/user turn pairs for context (optional).
            previous_turns = self._get_previous_turns(session)

            result = service.run(  # type: ignore[attr-defined]
                user_text=user_text,
                item_name=item_name,
                group=group,
                existing_selections=existing_selections,
                local_resolved=local_resolved,
                repeat_count=repeat_count,
                previous_turns=previous_turns,
                last_response_key=last_response_key,
                has_correction_signal=_has_correction,
            )

            # ── Structured logging for Phase 3 option resolver ────────────
            _logger.info(
                "option_resolver_result",
                extra={
                    "event": "option_resolver_result",
                    "group_id": group.group_id,
                    "group_name": group.name,
                    "item_name": item_name,
                    "option_resolver_mode": result.route_mode,
                    "option_resolver_route_reason": result.reason_code,
                    "option_resolver_called": result.gpt_called,
                    "option_resolver_decision": result.decision,
                    "option_resolver_selected_names": list(result.selected_names),
                    "option_resolver_confidence": result.confidence,
                    "option_resolver_safe_to_apply": result.safe_to_apply,
                    "option_resolver_applied": (
                        result.safe_to_apply
                        and result.decision == "select_option"
                        and bool(result.selected_names)
                    ),
                    "option_resolver_error": result.parse_error,
                    "option_resolver_skipped_reason": result.skipped_reason,
                    "option_resolver_latency_ms": result.latency_ms,
                    "repeat_loop_detected": repeat_count >= int(getattr(
                        getattr(service, "_config", None),
                        "option_resolver_repeat_threshold", 2
                    ) or 2),
                    "has_correction_signal": _has_correction,
                },
            )

            return result
        except Exception:
            return OPTION_RESOLVER_NOT_CALLED

    @staticmethod
    def _get_previous_turns(
        session: "Session | None",
    ) -> list[tuple[str, str]]:
        """Extract recent bot/user turn pairs from the session context."""
        try:
            if session is None:
                return []
            ctx = getattr(session, "conversation_context", None)
            if ctx is None:
                return []
            get_mem = getattr(ctx, "get_turn_memory", None)
            if not callable(get_mem):
                return []
            history = list(get_mem(3))
            result: list[tuple[str, str]] = []
            for turn in history:
                if isinstance(turn, (list, tuple)) and len(turn) == 2:
                    result.append((str(turn[0]), str(turn[1])))
            return result
        except Exception:
            return []

    # ------------------------------------------------------------------
    # SmartTurnPlanner helpers (modifier correction / low-confidence)
    # ------------------------------------------------------------------

    def _try_smart_planner_modifier(
        self,
        *,
        user_text: str,
        group: "PendingModifierGroup",
        pending: object,
        existing_selections: "list",
        context: "ConversationContext",
        session: object,
    ) -> "HandlerResult | None":
        """Try SmartTurnPlanner for correction phrases or low-confidence modifier turns.

        Returns a HandlerResult if the planner resolves a modifier, or None to
        continue to Phase 3 (GptOptionResolverService) then the local fallback.
        Never raises.
        """
        try:
            from app.services.smart_turn_planner import _is_enabled as _stp_enabled
            if not _stp_enabled():
                return None

            from app.services.smart_turn_policy import (
                should_use_smart_planner,
                determine_smart_task_mode,
                validate_smart_plan,
            )
            from app.services.smart_turn_planner import plan_smart_turn
            from app.services.smart_turn_context_builder import build_smart_turn_context

            local_confidence = float(context.last_intent_confidence or 0.0)
            state_name = ConversationState.WAITING_FOR_MODIFIER.value

            should_use, trigger_reason = should_use_smart_planner(
                transcript=user_text,
                state=state_name,
                local_intent="ADD_ITEM",
                local_confidence=local_confidence,
            )
            if not should_use:
                return None

            # Gather current group choices as allowed_options
            allowed_options = [choice.name for choice in group.choices]

            stp_ctx = build_smart_turn_context(
                transcript=user_text,
                state=state_name,
                local_intent="ADD_ITEM",
                local_confidence=local_confidence,
                context=context,
                session=session,
                allowed_options=allowed_options,
            )
            task_mode = determine_smart_task_mode(
                transcript=user_text,
                state=state_name,
                local_intent="ADD_ITEM",
                local_confidence=local_confidence,
            )

            pending_item_name = getattr(pending, "item_name", None)

            plan = plan_smart_turn(
                transcript=user_text,
                state=state_name,
                local_intent="ADD_ITEM",
                local_confidence=local_confidence,
                menu_context=[],
                cart_snapshot=stp_ctx.cart_snapshot,
                last_cart_diff=stp_ctx.last_cart_diff,
                previous_turns=stp_ctx.previous_turns,
                trigger_reason=trigger_reason,
                task_mode=task_mode,
                allowed_options=allowed_options,
                pending_item_name=pending_item_name,
                pending_group_name=group.name if hasattr(group, "name") else None,
                reprompt_count=stp_ctx.reprompt_count,
            )
            if plan is None:
                return None

            validation = validate_smart_plan(
                plan,
                # Pass empty menu_context so Gate 5 (item_name in menu) is
                # skipped for modifier turns.  The resolved modifier name is
                # validated separately via build_modifier_selections_from_names,
                # which checks group membership deterministically.
                menu_context=[],
                cart_snapshot=stp_ctx.cart_snapshot,
                state=state_name,
                local_intent="ADD_ITEM",
                trigger_reason=trigger_reason,
            )

            _logger.info(
                "smart_turn_planner_modifier",
                extra={
                    "smart_planner_invoked": True,
                    "smart_planner_task_mode": task_mode,
                    "smart_planner_trigger_reason": trigger_reason,
                    "smart_planner_decision": getattr(plan, "decision", ""),
                    "smart_planner_confidence": getattr(plan, "confidence", None),
                    "smart_planner_latency_ms": getattr(plan, "latency_ms", None),
                    "smart_planner_validation_result": validation.is_safe,
                    "smart_planner_fallback_reason": (
                        validation.block_reason if not validation.is_safe else None
                    ),
                    "smart_planner_context_keys": stp_ctx.context_keys,
                    "allowed_options_count": len(allowed_options),
                    "state_before": state_name,
                    "group_id": getattr(group, "group_id", ""),
                },
            )

            if not validation.is_safe:
                return None

            # Extract the resolved modifier name from the plan
            resolved_name = self._extract_modifier_name_from_plan(plan)
            if not resolved_name:
                return None

            # Validate through deterministic modifier path
            from app.nlu.semantic_repair.option_selection_validator import (
                build_modifier_selections_from_names,
            )
            gpt_selections = build_modifier_selections_from_names(
                selected_names=(resolved_name,),
                group=group,
                existing_ids={sel.modifier_id for sel in existing_selections},
            )
            if not gpt_selections:
                _logger.info(
                    "smart_turn_planner_modifier_name_not_in_group",
                    extra={"resolved_name": resolved_name, "group_id": getattr(group, "group_id", "")},
                )
                return None

            _logger.info(
                "smart_turn_planner_modifier_applied",
                extra={
                    "resolved_name": resolved_name,
                    "group_id": getattr(group, "group_id", ""),
                    "state_after": state_name,
                },
            )
            return self._apply_modifier_selection(
                context=context,
                pending=pending,
                group=group,
                matched_selections=gpt_selections,
                normalized_user_text=user_text,
            )
        except Exception as exc:
            _logger.warning("smart_turn_planner_modifier_error: %s", exc)
            return None

    @staticmethod
    def _extract_modifier_name_from_plan(plan: object) -> "str | None":
        """Extract the first resolved modifier name from a SmartTurnPlan.

        For modifier_selection task_mode the plan puts the resolved name in
        items[0].modifiers[0].name.  For correction task_mode the plan puts
        it in correction.corrected_text.  Returns None if neither is found.
        """
        decision = getattr(plan, "decision", "")

        if decision == "correction":
            corr = getattr(plan, "correction", None)
            if corr is not None:
                name = getattr(corr, "corrected_text", "").strip()
                return name or None

        if decision == "add_items":
            items = getattr(plan, "items", ()) or ()
            if items:
                mods = getattr(items[0], "modifiers", ()) or ()
                if mods:
                    name = getattr(mods[0], "name", "").strip()
                    return name or None

        return None

    # ------------------------------------------------------------------

    def _choice_payload(
        self,
        group: PendingModifierGroup,
        selections: list[ModifierSelection] | None = None,
    ) -> dict:
        selections = selections or []
        selected_ids = {sel.modifier_id for sel in selections}

        selected_names: list[str] = []
        for sel in selections:
            if sel.action == "remove":
                selected_names.append(f"no {sel.name}")
            elif sel.instruction == "extra":
                selected_names.append(f"extra {sel.name}")
            elif sel.instruction == "less":
                selected_names.append(f"less {sel.name}")
            else:
                selected_names.append(sel.name)

        selected_count = len(selections)
        min_selector, max_selector = effective_group_selector_bounds(group)
        remaining_choice_names = [
            choice.name
            for choice in group.choices
            if choice.modifier_id not in selected_ids
        ]

        return {
            "group_name": group.name,
            "top_choices": remaining_choice_names[:6],
            "all_choices": remaining_choice_names,
            "total_choices": len(group.choices),
            "selected_names": selected_names,
            "selected_count": selected_count,
            "min_selector": min_selector,
            "max_selector": max_selector,
            "remaining_to_min": max(min_selector - selected_count, 0),
            "remaining_to_max": max(max_selector - selected_count, 0),
        }

    @staticmethod
    def _all_modifier_choice_phrases(pending) -> list[str]:
        phrases: list[str] = []
        seen: set[str] = set()
        for group in pending.modifier_groups:
            for choice in group.choices:
                for value in getattr(choice, "match_texts", ()) or (choice.normalized_name,):
                    if value and value not in seen:
                        seen.add(value)
                        phrases.append(value)
        return phrases

    @staticmethod
    def _target_in_modifier_group(target: str, group: PendingModifierGroup) -> bool:
        """Return True if *target* token-overlaps any choice in *group*."""
        if not target:
            return False
        target_norm = _normalize_text(target)
        for choice in group.choices:
            choice_norm = getattr(choice, "normalized_name", None) or _normalize_text(choice.name)
            if target_norm == choice_norm or target_norm in choice_norm or choice_norm in target_norm:
                return True
            for match_text in getattr(choice, "match_texts", ()) or ():
                mt = _normalize_text(str(match_text))
                if mt and (target_norm == mt or target_norm in mt):
                    return True
        return False

    def _prefill_following_modifier_groups(
        self,
        *,
        context: ConversationContext,
        pending,
        start_index: int,
        normalized_user_text: str,
        consumed_values: list[str] | None = None,
    ) -> _PrefilledModifierGroups:
        carried_names: list[str] = []
        applied = False
        consumed = list(consumed_values or [])

        for later_index in range(start_index, len(pending.modifier_groups)):
            later_group = pending.modifier_groups[later_index]
            existing = list(context.selected_modifier_groups.get(later_group.group_id, []))
            existing_ids = [selection.modifier_id for selection in existing]
            consumed.extend(self._selected_choice_match_values(later_group, existing))

            resolution = self.modifier_resolver.resolve(
                group=later_group,
                normalized_user_text=normalized_user_text,
                option_candidates=build_modifier_option_candidates(context, normalized_user_text),
                normalized_slot_values=extract_modifier_slot_values_normalized(context),
                already_selected_ids=existing_ids,
                ignored_values=consumed,
                known_choice_phrases=self._all_modifier_choice_phrases(pending),
            )
            if not resolution.selections:
                continue

            proposed = list(existing)
            seen_ids = {selection.modifier_id for selection in existing}
            for selection in resolution.selections:
                if selection.modifier_id not in seen_ids:
                    proposed.append(selection)
                    seen_ids.add(selection.modifier_id)

            _, max_selector = effective_group_selector_bounds(later_group)
            if max_selector > 0 and len(proposed) > max_selector:
                requested_names = [
                    selection.name
                    for selection in proposed
                    if selection.modifier_id not in existing_ids
                ]
                return _PrefilledModifierGroups(
                    matched_names=carried_names,
                    applied=applied,
                    overflow_group_id=later_group.group_id,
                    overflow_requested_names=requested_names,
                    overflow_unmatched_names=[value for value in resolution.unmatched_values if value],
                    overflow_max_selector=max_selector,
                )

            accepted = proposed
            context.selected_modifier_groups[later_group.group_id] = accepted
            context.skipped_modifier_groups.discard(later_group.group_id)

            accepted_ids = {
                selection.modifier_id
                for selection in accepted
                if selection.modifier_id not in existing_ids
            }
            newly_added = [
                selection.name
                for selection in accepted
                if selection.modifier_id in accepted_ids
            ]
            if newly_added:
                carried_names.extend(newly_added)
                applied = True
                consumed.extend(self._selected_choice_match_values(later_group, accepted))

        return _PrefilledModifierGroups(
            matched_names=carried_names,
            applied=applied,
        )

    @staticmethod
    def _selected_choice_match_values(
        group: PendingModifierGroup,
        selections: list[ModifierSelection],
    ) -> list[str]:
        values: list[str] = []
        seen: set[str] = set()
        for selection in selections:
            choice = group.choices_by_modifier_id.get(selection.modifier_id)
            candidates = getattr(choice, "match_texts", ()) if choice is not None else ()
            for value in (*candidates, selection.name.lower()):
                normalized = str(value or "").strip()
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    values.append(normalized)
        return values

