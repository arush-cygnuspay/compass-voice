# app/state_machine/handlers/item/add_item/waiting_for_side_handler.py
from __future__ import annotations

from dataclasses import dataclass

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
from app.state_machine.models.pending_item_models import InterruptProposal, PendingSideGroup
from app.state_machine.models.conversation_state import ConversationState
from app.state_machine.handler_result import HandlerResult
from app.state_machine.handlers.item.add_item.group_resolution_handler import GroupResolutionHandler
from app.state_machine.handlers.item.add_item.add_item_handler import PendingItemCaptureHelper
from app.state_machine.handlers.item.add_item.add_item_flow import (
    ReadyToFinalize,
    determine_next_add_item_step,
)
from app.state_machine.handlers.item.add_item.side_group_resolver import (
    SideGroupResolver,
    build_side_option_candidates,
    extract_side_slot_values_normalized,
    dedupe_keep_order,
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
from app.utils.token_matcher import tokenize
from app.nlu.control_phrase_classifier import DEFAULT_CLASSIFIER
from app.nlu.utterance_filter import DEFAULT_FILTER


@dataclass(frozen=True, slots=True)
class _ScoredSideChoice:
    item_id: str
    choice_name: str
    confidence: float


def _looks_like_done_answer(normalized_user_text: str) -> bool:
    return looks_like_done_answer(normalized_user_text)


def _looks_like_more_options(normalized_user_text: str) -> bool:
    return looks_like_more_options_answer(normalized_user_text)


def _looks_like_skip_side_answer(normalized_user_text: str, group: PendingSideGroup) -> bool:
    text = (normalized_user_text or "").strip()
    if not text:
        return False

    return looks_like_skip_answer(text)


class WaitingForSideHandler(GroupResolutionHandler):
    """
    Resolve side selections strictly from the active pending item snapshot.

    Important:
    - no broad menu resolution during waiting state
    - only choices from the current active side group can match
    - supports multi-side capture in one utterance
    - keeps group open when min is met but more are still allowed
    - preserves existing side-size handoff for newly selected variant-priced sides
    """

    def __init__(self, menu_repo: MenuRepository | None = None) -> None:
        self.menu_repo = menu_repo
        self.side_resolver = SideGroupResolver()
        self.capture_helper = PendingItemCaptureHelper(side_resolver=self.side_resolver)

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
        groups = pending.side_groups
        idx = context.current_side_group_index

        if idx >= len(groups):
            step = determine_next_add_item_step(context)
            return self._step_to_result(context, step)

        group = groups[idx]
        existing_ids = list(context.selected_side_groups.get(group.group_id, []))
        min_selector, max_selector = effective_group_selector_bounds(group)
        self.capture_helper.prefill_quantity(
            context=context,
            user_text=normalized_user_text,
        )

        # ── Control-phrase pre-classification ────────────────────────────
        # Intercept skip/done/repeat/negated_option BEFORE the resolver so
        # phrases like "no skip that", "can you repeat", "no bun" (on a
        # required group) never reach unmatched_names and get echoed back.
        _cp = DEFAULT_CLASSIFIER.classify(
            normalized_user_text, ConversationState.WAITING_FOR_SIDE.value
        )
        if _cp.action == "repeat":
            log_control_intent_event(
                "control_phrase_classifier_repeat",
                state=ConversationState.WAITING_FOR_SIDE.value,
                group_id=group.group_id,
            )
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_SIDE,
                response_key="repeat_side_options",
                response_payload={
                    **self._choice_payload(context, group),
                    "repeat_reason": "meta_clarify",
                },
            )

        if _cp.action in {"skip", "done"}:
            log_control_intent_event(
                "control_phrase_classifier_skip_done",
                state=ConversationState.WAITING_FOR_SIDE.value,
                action=_cp.action,
                group_id=group.group_id,
            )
            _skip = evaluate_group_skip(min_selector, len(existing_ids))
            if _skip.decision == GroupSkipDecision.BLOCK_UNDER_MIN:
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_SIDE,
                    response_key="required_side_cannot_skip",
                    response_payload={
                        **self._choice_payload(context, group),
                        "remaining_to_min": _skip.remaining_to_min,
                        "selected_count": _skip.selected_count,
                        "min_required": _skip.min_required,
                        "intent_kind": _cp.action,
                    },
                )
            if _skip.decision == GroupSkipDecision.SKIP_OPTIONAL and not existing_ids:
                context.skipped_side_groups.add(group.group_id)
                context.selected_side_groups.pop(group.group_id, None)
            elif _skip.decision == GroupSkipDecision.ADVANCE_MIN_MET:
                log_control_intent_event(
                    "advance_min_met",
                    state=ConversationState.WAITING_FOR_SIDE.value,
                    field_name="side",
                    group_id=group.group_id,
                    selected_count=_skip.selected_count,
                    min_required=_skip.min_required,
                    kind=_cp.action,
                )
            step = determine_next_add_item_step(context)
            return self._step_to_result(context, step)

        if _cp.action == "negated_option":
            # In a side group, "no X" means the user does not want any
            # side matching X.  Side groups require positive selections, so
            # we either block (required) or skip (optional).
            log_control_intent_event(
                "control_phrase_classifier_negated_option",
                state=ConversationState.WAITING_FOR_SIDE.value,
                target=_cp.normalized_target,
                group_id=group.group_id,
            )
            _skip = evaluate_group_skip(min_selector, len(existing_ids))
            if _skip.decision == GroupSkipDecision.BLOCK_UNDER_MIN:
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_SIDE,
                    response_key="required_side_cannot_skip",
                    response_payload={
                        **self._choice_payload(context, group),
                        "remaining_to_min": _skip.remaining_to_min,
                        "selected_count": _skip.selected_count,
                        "min_required": _skip.min_required,
                        "intent_kind": "negated_option",
                    },
                )
            if not existing_ids:
                context.skipped_side_groups.add(group.group_id)
                context.selected_side_groups.pop(group.group_id, None)
            step = determine_next_add_item_step(context)
            return self._step_to_result(context, step)
        # ── END control-phrase pre-classification ─────────────────────────

        control_intent = resolve_control_intent(
            normalized_user_text,
            intent,
            getattr(context.last_nlu, "model_sub_intent", None),
            ConversationState.WAITING_FOR_SIDE,
            context,
            nlu_result=context.last_nlu,
            intent_confidence=context.last_intent_confidence,
        )

        if control_intent is not None:
            if control_intent.kind == ControlIntentKind.OPTIONS_REQUEST:
                log_control_intent_event(
                    "options_requested",
                    state=ConversationState.WAITING_FOR_SIDE.value,
                    field_name="side",
                    group_id=group.group_id,
                )
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_SIDE,
                    response_key="list_side_options",
                    response_payload=self._choice_payload(context, group),
                )

            if control_intent.kind == ControlIntentKind.META_CLARIFY:
                log_control_intent_event(
                    "meta_clarify_repeated",
                    state=ConversationState.WAITING_FOR_SIDE.value,
                    field_name="side",
                    group_id=group.group_id,
                )
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_SIDE,
                    response_key="repeat_side_options",
                    response_payload={
                        **self._choice_payload(context, group),
                        "repeat_reason": "meta_clarify",
                    },
                )

            if control_intent.kind == ControlIntentKind.CANCEL:
                log_control_intent_event(
                    "control_intent_action",
                    state=ConversationState.WAITING_FOR_SIDE.value,
                    action="cancel_pending_item",
                    kind=control_intent.kind.value,
                )
                context.reset_item_scope()
                return HandlerResult(
                    next_state=ConversationState.IDLE,
                    response_key="item_cancelled_successfully",
                )

            if control_intent.kind == ControlIntentKind.AFFIRM:
                if len(existing_ids) >= min_selector and existing_ids:
                    log_control_intent_event(
                        "control_intent_action",
                        state=ConversationState.WAITING_FOR_SIDE.value,
                        action="accept_current_side_selection",
                        kind=control_intent.kind.value,
                    )
                    step = determine_next_add_item_step(context)
                    return self._step_to_result(context, step)

                log_control_intent_event(
                    "control_intent_action",
                    state=ConversationState.WAITING_FOR_SIDE.value,
                    action="side_selection_requires_explicit_choice",
                    kind=control_intent.kind.value,
                )
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_SIDE,
                    response_key="repeat_side_options",
                    response_payload={
                        **self._choice_payload(context, group),
                        "repeat_reason": "need_choice",
                    },
                )

            if control_intent.kind in {ControlIntentKind.DENY, ControlIntentKind.DONE}:
                skip = evaluate_group_skip(min_selector, len(existing_ids))

                if skip.decision == GroupSkipDecision.BLOCK_UNDER_MIN:
                    log_control_intent_event(
                        "required_selection_cannot_skip",
                        state=ConversationState.WAITING_FOR_SIDE.value,
                        field_name="side",
                        group_id=group.group_id,
                    )
                    return HandlerResult(
                        next_state=ConversationState.WAITING_FOR_SIDE,
                        response_key="required_side_cannot_skip",
                        response_payload={
                            **self._choice_payload(context, group),
                            "remaining_to_min": skip.remaining_to_min,
                            "selected_count": skip.selected_count,
                            "min_required": skip.min_required,
                            "intent_kind": control_intent.kind.value,
                        },
                    )

                if skip.decision == GroupSkipDecision.SKIP_OPTIONAL:
                    # Preserve byte-identical legacy behavior: only mark
                    # the group as "skipped" when the user actually had
                    # no selections. When selections exist (min == 0
                    # with prior picks) we just advance with picks intact.
                    if not existing_ids:
                        context.skipped_side_groups.add(group.group_id)
                        context.selected_side_groups.pop(group.group_id, None)
                        log_control_intent_event(
                            "skipped_optional_group",
                            state=ConversationState.WAITING_FOR_SIDE.value,
                            field_name="side",
                            group_id=group.group_id,
                        )
                else:
                    # ADVANCE_MIN_MET: selections meet/exceed min — keep
                    # them, do not flag as skipped.
                    log_control_intent_event(
                        "advance_min_met",
                        state=ConversationState.WAITING_FOR_SIDE.value,
                        field_name="side",
                        group_id=group.group_id,
                        selected_count=skip.selected_count,
                        min_required=skip.min_required,
                        kind=control_intent.kind.value,
                    )

                step = determine_next_add_item_step(context)
                return self._step_to_result(context, step)

        resolution = self.side_resolver.resolve(
            group=group,
            normalized_user_text=normalized_user_text,
            option_candidates=build_side_option_candidates(context, normalized_user_text),
            normalized_slot_values=extract_side_slot_values_normalized(context),
            already_selected_ids=existing_ids,
        )

        if resolution.matched_item_ids:
            return self._apply_side_selection(
                context=context,
                pending_item_name=pending.item_name,
                group=group,
                matched_ids=resolution.matched_item_ids,
                unmatched_values=resolution.unmatched_values,
                normalized_user_text=normalized_user_text,
                match_debug=resolution.match_debug,
            )

        carry_feedback = self.capture_helper.prefill_side_groups(
            context=context,
            normalized_user_text=normalized_user_text,
            start_index=idx + 1,
        )
        self.capture_helper.prefill_selected_side_variants(
            context=context,
            user_text=normalized_user_text,
            slots=context.last_slots or (),
        )
        modifier_feedback = self.capture_helper.prefill_modifier_groups(
            context=context,
            normalized_user_text=normalized_user_text,
        )
        carried_names = self.capture_helper.collect_matched_names(carry_feedback + modifier_feedback)
        filtered_unmatched = DEFAULT_FILTER.strip_unmatched(
            self._filter_unmatched_values(
                resolution.unmatched_values,
                matched_names=carried_names,
            )
        )

        if carried_names and len(existing_ids) >= min_selector:
            if not existing_ids:
                context.skipped_side_groups.add(group.group_id)
                context.selected_side_groups.pop(group.group_id, None)

            step = determine_next_add_item_step(context)
            return self._step_to_result(
                context,
                step,
                matched_names=carried_names,
                unmatched_names=filtered_unmatched,
            )

        if resolution.unmatched_values:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_SIDE,
                response_key="repeat_side_options",
                response_payload={
                    **self._choice_payload(context, group),
                    "repeat_reason": "invalid",
                    "matched_names": carried_names,
                    "unmatched_names": filtered_unmatched,
                    **self._match_debug_payload(resolution.match_debug),
                },
            )

        if intent in SOFT_SWITCH_INTENTS:
            context.return_state = ConversationState.WAITING_FOR_SIDE
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
            next_state=ConversationState.WAITING_FOR_SIDE,
            response_key="repeat_side_options",
            response_payload={
                **self._choice_payload(context, group),
                "repeat_reason": "invalid",
                "matched_names": carried_names,
                **self._match_debug_payload(resolution.match_debug),
            },
        )

    def _apply_side_selection(
        self,
        *,
        context: ConversationContext,
        pending_item_name: str,
        group: PendingSideGroup,
        matched_ids: list[str],
        unmatched_values: list[str] | None = None,
        normalized_user_text: str,
        match_debug: dict[str, object] | None = None,
    ) -> HandlerResult:
        existing_ids = list(context.selected_side_groups.get(group.group_id, []))
        proposed_ids = dedupe_keep_order(existing_ids + matched_ids)

        min_selector, max_selector = effective_group_selector_bounds(group)

        # Build feedback lists
        _unmatched = [v for v in (unmatched_values or []) if v]

        # ── Over-max: accept up to limit, tell user what was capped ──
        if max_selector > 0 and len(proposed_ids) > max_selector:
            dropped_ids = proposed_ids[max_selector:]
            dropped_names = [
                group.choices_by_item_id[item_id].name
                for item_id in dropped_ids
                if item_id in group.choices_by_item_id
            ]
            requested_names = [
                group.choices_by_item_id[item_id].name
                for item_id in matched_ids
                if item_id in group.choices_by_item_id
            ]

            payload = self._choice_payload(context, group)
            payload["requested_names"] = requested_names
            payload["dropped_names"] = dropped_names
            payload["unmatched_names"] = _unmatched
            payload["over_max"] = True
            payload.update(self._match_debug_payload(match_debug))
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_SIDE,
                response_key="too_many_side_choices",
                response_payload=payload,
            )

        context.selected_side_groups[group.group_id] = proposed_ids
        context.skipped_side_groups.discard(group.group_id)

        newly_added_ids = [item_id for item_id in proposed_ids if item_id not in existing_ids]
        newly_added_names = [
            group.choices_by_item_id[item_id].name
            for item_id in newly_added_ids
            if item_id in group.choices_by_item_id
        ]

        carry_feedback = self.capture_helper.prefill_side_groups(
            context=context,
            normalized_user_text=normalized_user_text,
            start_index=context.current_side_group_index + 1,
        )
        self.capture_helper.prefill_selected_side_variants(
            context=context,
            user_text=normalized_user_text,
            slots=context.last_slots or (),
        )
        modifier_feedback = self.capture_helper.prefill_modifier_groups(
            context=context,
            normalized_user_text=normalized_user_text,
        )
        all_matched_names = newly_added_names + self.capture_helper.collect_matched_names(
            carry_feedback + modifier_feedback
        )

        for selected_item_id in newly_added_ids:
            choice = group.choices_by_item_id.get(selected_item_id)
            if choice and choice.pricing_mode == "variant":
                if selected_item_id in context.selected_side_variants:
                    continue
                context.pending_side_item_id = choice.item_id
                context.pending_side_item_name = choice.name
                context.pending_side_group_id = group.group_id
                context.current_prompt_field = "side_size"
                context.available_choices_kind = "side_size"
                context.available_choices_values = choice.variant_names

                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_SIDE_SIZE,
                    response_key="ask_for_side_size",
                    response_payload={
                        "item_name": pending_item_name,
                        "side_item_name": choice.name,
                        "group_name": group.name,
                        "available_sizes": list(choice.variant_names),
                        "matched_names": all_matched_names,
                        "unmatched_names": _unmatched,
                    },
                )

        if len(proposed_ids) < min_selector:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_SIDE,
                response_key="repeat_side_options",
                response_payload={
                    **self._choice_payload(context, group),
                    "repeat_reason": "need_more",
                    "matched_names": all_matched_names,
                    "unmatched_names": _unmatched,
                    **self._match_debug_payload(match_debug),
                },
            )

        step = determine_next_add_item_step(context)
        return self._step_to_result(
            context, step,
            matched_names=all_matched_names,
            unmatched_names=_unmatched,
            match_debug=match_debug,
        )

    def _choice_payload(self, context: ConversationContext, group: PendingSideGroup) -> dict:
        selected_ids = list(context.selected_side_groups.get(group.group_id, []))
        selected_names = [
            group.choices_by_item_id[item_id].name
            for item_id in selected_ids
            if item_id in group.choices_by_item_id
        ]

        selected_count = len(selected_ids)
        min_selector, max_selector = effective_group_selector_bounds(group)
        selected_id_set = set(selected_ids)
        remaining_choice_names = [
            choice.name
            for choice in group.choices
            if choice.item_id not in selected_id_set
        ]

        return {
            "group_name": group.name,
            "top_choices": remaining_choice_names[:4],
            "all_choices": remaining_choice_names,
            "selected_names": selected_names,
            "selected_count": selected_count,
            "min_selector": min_selector,
            "max_selector": max_selector,
            "remaining_to_min": max(min_selector - selected_count, 0),
            "remaining_to_max": max(max_selector - selected_count, 0),
        }

    @staticmethod
    def _filter_unmatched_values(
        unmatched_values: list[str] | None,
        *,
        matched_names: list[str] | None = None,
    ) -> list[str]:
        values = [str(value).strip() for value in (unmatched_values or []) if str(value).strip()]
        if not values:
            return []

        matched_tokens: set[str] = set()
        for name in matched_names or []:
            matched_tokens.update(tokenize(normalize_text(name)))

        if not matched_tokens:
            return values

        filtered: list[str] = []
        for value in values:
            value_tokens = set(tokenize(normalize_text(value)))
            if value_tokens and value_tokens.issubset(matched_tokens):
                continue
            filtered.append(value)
        return filtered

