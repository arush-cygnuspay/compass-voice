# app/state_machine/handlers/item/add_item/waiting_for_modifier_handler.py
from __future__ import annotations

from dataclasses import dataclass

from app.menu.repository import MenuRepository
from app.nlu.intent_resolution.intent import Intent
from app.session.session import Session
from app.state_machine.control_intent_resolver import (
    ControlIntentKind,
    log_control_intent_event,
    resolve_control_intent,
)
from app.state_machine.models.conversation_context import (
    ConversationContext,
    InterruptProposal,
    PendingModifierGroup,
)
from app.state_machine.models.conversation_state import ConversationState
from app.state_machine.models.pending_item_models import ModifierSelection
from app.state_machine.handler_result import HandlerResult
from app.state_machine.handlers.base_handler import BaseHandler
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


class WaitingForModifierHandler(BaseHandler):
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
                context.reset_task()
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
                    response_key="repeat_modifier_options",
                    response_payload={
                        **self._choice_payload(group, existing_selections),
                        "repeat_reason": "need_more",
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

        if resolution.unmatched_values:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_MODIFIER,
                response_key="repeat_modifier_options",
                response_payload={
                    **self._choice_payload(group, existing_selections),
                    "repeat_reason": "invalid",
                    "unmatched_names": [value for value in resolution.unmatched_values if value],
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
                response_key="repeat_modifier_options",
                response_payload={
                    **self._choice_payload(group, proposed),
                    "repeat_reason": "need_more",
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
            "top_choices": remaining_choice_names[:4],
            "all_choices": remaining_choice_names,
            "selected_names": selected_names,
            "selected_count": selected_count,
            "min_selector": min_selector,
            "max_selector": max_selector,
            "remaining_to_min": max(min_selector - selected_count, 0),
            "remaining_to_max": max(max_selector - selected_count, 0),
        }

    def _step_to_result(
        self,
        context: ConversationContext,
        step,
        *,
        matched_names: list[str] | None = None,
        unmatched_names: list[str] | None = None,
        match_debug: dict[str, object] | None = None,
    ) -> HandlerResult:
        pending = context.pending_add_item
        if pending is None:
            return HandlerResult(
                next_state=ConversationState.ERROR_RECOVERY,
                response_key="item_context_missing",
            )

        if isinstance(step, ReadyToFinalize):
            payload = {
                "item_name": pending.item_name,
                "quantity": context.quantity or 1,
            }
            if matched_names:
                payload["matched_names"] = matched_names
            if unmatched_names:
                payload["unmatched_names"] = unmatched_names
            payload.update(self._match_debug_payload(match_debug))
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="item_added_successfully",
                response_payload=payload,
                command=step.command.to_dict(),
                reset_context=True,
            )

        payload = step.response_payload or {}
        if matched_names:
            payload["matched_names"] = matched_names
        if unmatched_names:
            payload["unmatched_names"] = unmatched_names
        payload.update(self._match_debug_payload(match_debug))
        return HandlerResult(
            next_state=step.next_state,
            response_key=step.response_key,
            response_payload=payload,
        )

    @staticmethod
    def _match_debug_payload(match_debug: dict[str, object] | None) -> dict[str, object]:
        return dict(match_debug or {})

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

