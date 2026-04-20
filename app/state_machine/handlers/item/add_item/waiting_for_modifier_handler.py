# app/state_machine/handlers/item/add_item/waiting_for_modifier_handler.py
from __future__ import annotations

from app.menu.repository import MenuRepository
from app.nlu.intent_resolution.intent import Intent
from app.session.session import Session
from app.state_machine.models.conversation_context import (
    ConversationContext,
    InterruptProposal,
    PendingModifierGroup,
)
from app.state_machine.models.conversation_state import ConversationState
from app.state_machine.models.pending_item_models import ModifierSelection
from app.state_machine.handler_result import HandlerResult
from app.state_machine.handlers.base_handler import BaseHandler
from app.state_machine.handlers.item.add_item.add_item_flow import (
    build_add_item_command,
    determine_next_add_item_step,
)
from app.state_machine.handlers.item.add_item.modifier_group_resolver import (
    ModifierGroupResolver,
    extract_modifier_slot_values_normalized,
)
from app.state_machine.handlers.item.add_item.group_collection_utils import (
    effective_group_selector_bounds,
)

SOFT_SWITCH_INTENTS: set[Intent] = {
    Intent.ADD_ITEM,
    Intent.REMOVE_ITEM,
    Intent.MODIFY_ITEM,
    Intent.SHOW_MENU,
    Intent.ASK_MENU_INFO,
    Intent.ASK_PRICE,
    Intent.SHOW_CART,
    Intent.SHOW_TOTAL,
    Intent.START_ORDER,
    Intent.PAYMENT_REQUEST,
    Intent.CANCEL_ORDER,
    # NOTE: END_ADDING, CHECKOUT, FINISH_ORDER, CONFIRM_ORDER, REVIEW_ORDER
    # are handled by GROUP_DONE_INTENTS — they mean "done with this group"
    # in the side/modifier context, not "interrupt the current item".
}

DONE_WORDS = {
    "done",
    "thats all",
    "that's all",
    "thats it",
    "that's it",
    "finished",
    "continue",
    "next",
    "no more",
    "nothing else",
    "i'm good",
    "im good",
    "i dont want anymore",
    "i don't want anymore",
    "i dont want any more",
    "i don't want any more",
    "thats enough",
    "that's enough",
    "i'm done",
    "im done",
    "all good",
    "good",
    "nah thats it",
    "nah that's it",
}

SKIP_WORDS = {
    "no",
    "none",
    "nothing",
    "skip",
    "skip it",
    "no thanks",
}

MORE_OPTIONS_WORDS = {
    "other options",
    "more options",
    "what else",
    "what else do you have",
    "what else you got",
    "next options",
    "show me more",
    "any others",
    "anything else available",
    "what are my options",
    "options",
}

# Intents that mean "I'm done ordering" but in side/modifier context
# should be treated as "done with this group" — NOT as a flow interruption.
GROUP_DONE_INTENTS: set[Intent] = {
    Intent.END_ADDING,
    Intent.CHECKOUT,
    Intent.FINISH_ORDER,
    Intent.CONFIRM_ORDER,
    Intent.REVIEW_ORDER,
}


def _looks_like_done_answer(normalized_user_text: str) -> bool:
    return (normalized_user_text or "").strip() in DONE_WORDS


def _looks_like_more_options(normalized_user_text: str) -> bool:
    text = (normalized_user_text or "").strip()
    return text in MORE_OPTIONS_WORDS


def _looks_like_skip_modifier_answer(normalized_user_text: str, group: PendingModifierGroup) -> bool:
    text = (normalized_user_text or "").strip()
    if not text:
        return False

    # whole-group skip only; specific "no onions" is handled by the resolver
    return text in SKIP_WORDS


def _looks_like_specific_modifier_removal(normalized_user_text: str) -> bool:
    text = (normalized_user_text or "").strip()
    return bool(text) and (text.startswith("no ") or text.startswith("without "))


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

        min_selector, max_selector = effective_group_selector_bounds(group)

        # ── "What else?" / "more options" → show option listing ──
        if intent == Intent.ASK_OPTIONS or _looks_like_more_options(normalized_user_text):
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_MODIFIER,
                response_key="list_modifier_options",
                response_payload=self._choice_payload(group, existing_selections),
            )

        # ── Skip / deny → skip group if optional ──
        if (
            (intent == Intent.DENY and not _looks_like_specific_modifier_removal(normalized_user_text))
            or _looks_like_skip_modifier_answer(normalized_user_text, group)
        ):
            if len(existing_selections) < min_selector:
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_MODIFIER,
                    response_key="required_modifier_cannot_skip",
                    response_payload=self._choice_payload(group, existing_selections),
                )

            if not existing_selections:
                context.skipped_modifier_groups.add(group.group_id)
                context.selected_modifier_groups.pop(group.group_id, None)

            step = determine_next_add_item_step(context)
            return self._step_to_result(context, step)

        # ── "That's it" / "done" / "no more" → done with THIS GROUP ──
        if _looks_like_done_answer(normalized_user_text) or (
            intent in GROUP_DONE_INTENTS
        ):
            if len(existing_selections) < min_selector:
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_MODIFIER,
                    response_key="repeat_modifier_options",
                    response_payload={
                        **self._choice_payload(group, existing_selections),
                        "repeat_reason": "need_more",
                    },
                )

            step = determine_next_add_item_step(context)
            return self._step_to_result(context, step)

        resolution = self.modifier_resolver.resolve(
            group=group,
            normalized_user_text=normalized_user_text,
            normalized_slot_values=extract_modifier_slot_values_normalized(context),
            already_selected_ids=existing_ids,
        )

        if resolution.selections:
            return self._apply_modifier_selection(
                context=context,
                group=group,
                matched_selections=resolution.selections,
                unmatched_values=resolution.unmatched_values,
            )

        if intent in SOFT_SWITCH_INTENTS:
            context.awaiting_flow_confirmation = True
            context.return_state = ConversationState.WAITING_FOR_MODIFIER
            context.interrupt_proposal = InterruptProposal(
                text=normalized_user_text,
                predicted_main_intent=None,
                predicted_sub_intent=intent.value,
            )
            return HandlerResult(
                next_state=ConversationState.CANCELLATION_CONFIRMATION,
                response_key="confirm_cancel_current_item_for_new_request",
                response_payload={"item_name": pending.item_name},
            )

        return HandlerResult(
            next_state=ConversationState.WAITING_FOR_MODIFIER,
            response_key="repeat_modifier_options",
            response_payload={
                **self._choice_payload(group, existing_selections),
                "repeat_reason": "invalid",
            },
        )

    def _apply_modifier_selection(
        self,
        *,
        context: ConversationContext,
        group: PendingModifierGroup,
        matched_selections: list[ModifierSelection],
        unmatched_values: list[str] | None = None,
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
            accepted = proposed[:max_selector]
            dropped = proposed[max_selector:]
            accepted_names = [sel.name for sel in accepted if sel not in existing]
            dropped_names = [sel.name for sel in dropped]

            context.selected_modifier_groups[group.group_id] = accepted
            context.skipped_modifier_groups.discard(group.group_id)

            payload = self._choice_payload(group, accepted)
            payload["accepted_names"] = accepted_names
            payload["dropped_names"] = dropped_names
            payload["unmatched_names"] = _unmatched
            payload["over_max"] = True
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_MODIFIER,
                response_key="too_many_modifier_choices",
                response_payload=payload,
            )

        context.selected_modifier_groups[group.group_id] = proposed
        context.skipped_modifier_groups.discard(group.group_id)

        if len(proposed) < min_selector:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_MODIFIER,
                response_key="repeat_modifier_options",
                response_payload={
                    **self._choice_payload(group, proposed),
                    "repeat_reason": "need_more",
                    "matched_names": newly_added_names,
                    "unmatched_names": _unmatched,
                },
            )

        if max_selector > 1 and len(proposed) < max_selector:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_MODIFIER,
                response_key="repeat_modifier_options",
                response_payload={
                    **self._choice_payload(group, proposed),
                    "repeat_reason": "optional_more",
                    "matched_names": newly_added_names,
                    "unmatched_names": _unmatched,
                },
            )

        step = determine_next_add_item_step(context)
        return self._step_to_result(
            context, step,
            matched_names=newly_added_names,
            unmatched_names=_unmatched,
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
    ) -> HandlerResult:
        pending = context.pending_add_item
        if pending is None:
            return HandlerResult(
                next_state=ConversationState.ERROR_RECOVERY,
                response_key="item_context_missing",
            )

        if step.next_state == ConversationState.FINALIZING_ADD_ITEM:
            payload = {
                "item_name": pending.item_name,
                "quantity": context.quantity or 1,
            }
            if matched_names:
                payload["matched_names"] = matched_names
            if unmatched_names:
                payload["unmatched_names"] = unmatched_names
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="item_added_successfully",
                response_payload=payload,
                command=build_add_item_command(context),
                reset_context=True,
            )

        payload = step.response_payload or {}
        if matched_names:
            payload["matched_names"] = matched_names
        if unmatched_names:
            payload["unmatched_names"] = unmatched_names
        return HandlerResult(
            next_state=step.next_state,
            response_key=step.response_key,
            response_payload=payload,
        )
