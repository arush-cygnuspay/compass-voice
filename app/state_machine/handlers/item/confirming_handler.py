# app/state_machine/handlers/item/confirming_handler.py

from __future__ import annotations

from dataclasses import dataclass

from app.core.pending_action import PendingAction
from app.menu.models import MenuItem
from app.menu.query_result import MenuQueryType
from app.menu.repository import MenuRepository
from app.nlu.intent_resolution.intent import Intent
from app.session.session import Session
from app.state_machine.handlers.base_handler import BaseHandler
from app.state_machine.conversation_context import ConversationContext, InterruptProposal
from app.state_machine.conversation_state import ConversationState
from app.state_machine.handler_result import HandlerResult
from app.state_machine.handlers.item.add_item.add_item_flow import (
    build_add_item_command,
    determine_next_add_item_step,
)
from app.state_machine.handlers.item.add_item.pending_add_item_factory import (
    build_pending_add_item,
)
from app.nlu.query_normalization.text_preprocessor import normalize_text


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
    Intent.END_ADDING,
    Intent.CHECKOUT,
    Intent.CONFIRM_ORDER,
    Intent.FINISH_ORDER,
    Intent.REVIEW_ORDER,
    Intent.PAYMENT_REQUEST,
    Intent.CANCEL_ORDER,
}


FILLER_PREFIXES: tuple[str, ...] = (
    "i want ",
    "i want a ",
    "i want an ",
    "i would like ",
    "i would like a ",
    "i would like an ",
    "add ",
    "get ",
    "give me ",
    "make it ",
    "send ",
    "bring ",
)


@dataclass(frozen=True, slots=True)
class _CandidateMatch:
    item: MenuItem
    match_type: str  # exact | strong


class ConfirmingHandler(BaseHandler):
    """
    Handles item-level confirmation only.

    Confirmation flow:
    1) ambiguous/category prompt offers candidate list
    2) user names one candidate -> assistant asks final yes/no confirm
    3) yes -> enter add flow
    4) no -> return to previous ambiguous/category prompt
    """

    def __init__(self, menu_repo: MenuRepository) -> None:
        self.menu_repo = menu_repo

    def handle(
            self,
            intent: Intent,
            context: ConversationContext,
            user_text: str,
            session: Session | None = None,
    ) -> HandlerResult:
        if session is None or session.conversation_state != ConversationState.CONFIRMING_ITEM:
            return HandlerResult(
                next_state=ConversationState.ERROR_RECOVERY,
                response_key="confirmation_state_error",
            )

        confirmation = context.awaiting_confirmation_for
        if not confirmation or confirmation.get("type") != "item":
            return HandlerResult(
                next_state=ConversationState.ERROR_RECOVERY,
                response_key="confirmation_state_error",
            )

        reason = confirmation.get("reason")

        # -------------------------------------------------
        # Stage A: unresolved ambiguous/category item prompt
        # -------------------------------------------------
        if reason in {"multiple_matches", "category_detected"}:
            # 1) First try matching against currently offered candidates
            matched = self._resolve_candidate_item_from_confirmation(
                context=context,
                user_text=user_text,
            )
            if matched is not None:
                return self._move_to_selected_candidate_state(
                    context=context,
                    item=matched.item,
                    previous_confirmation=confirmation,
                )

            # 2) Then try resolving the utterance as a fresh item request
            fresh_resolution = self._resolve_fresh_item_attempt(
                context=context,
                user_text=user_text,
            )
            if fresh_resolution is not None:
                return fresh_resolution

            # 3) Explicit deny means abandon clarification
            if intent == Intent.DENY:
                context.awaiting_confirmation_for = None
                context.candidate_item_id = None
                context.current_item_id = None
                context.current_item_name = None

                return HandlerResult(
                    next_state=ConversationState.IDLE,
                    response_key="item_cancelled_successfully",
                )

            # 4) Otherwise repeat the ambiguity prompt
            return HandlerResult(
                next_state=ConversationState.CONFIRMING_ITEM,
                response_key=self._repeat_response_key(confirmation),
                response_payload=dict(confirmation),
            )

        # -------------------------------------------------
        # Stage B: explicit yes/no for a concrete selected item
        # -------------------------------------------------
        if reason == "candidate_selected":
            if intent == Intent.CONFIRM:
                candidate_item_id = confirmation.get("value_id")
                if not candidate_item_id:
                    return HandlerResult(
                        next_state=ConversationState.ERROR_RECOVERY,
                        response_key="confirmation_state_error",
                    )

                try:
                    item = self.menu_repo.get_item(candidate_item_id)
                except KeyError:
                    return HandlerResult(
                        next_state=ConversationState.ERROR_RECOVERY,
                        response_key="item_context_missing",
                    )

                return self._enter_add_flow_for_item(
                    context=context,
                    item=item,
                )

            if intent == Intent.DENY:
                previous_confirmation = confirmation.get("previous_confirmation") or {}
                if not previous_confirmation:
                    context.awaiting_confirmation_for = None
                    context.candidate_item_id = None
                    return HandlerResult(
                        next_state=ConversationState.IDLE,
                        response_key="item_cancelled_successfully",
                    )

                context.candidate_item_id = None
                context.current_item_name = None
                context.awaiting_confirmation_for = dict(previous_confirmation)

                return HandlerResult(
                    next_state=ConversationState.CONFIRMING_ITEM,
                    response_key=self._repeat_response_key(previous_confirmation),
                    response_payload=dict(previous_confirmation),
                )

            # In candidate_selected stage, now soft-switch makes sense
            if intent in SOFT_SWITCH_INTENTS:
                item_name = confirmation.get("value_name") or "this item"

                context.awaiting_flow_confirmation = True
                context.return_state = ConversationState.CONFIRMING_ITEM
                context.interrupt_proposal = InterruptProposal(
                    text=user_text,
                    predicted_main_intent=None,
                    predicted_sub_intent=intent.value,
                )

                return HandlerResult(
                    next_state=ConversationState.CANCELLATION_CONFIRMATION,
                    response_key="confirm_cancel_current_item_for_new_request",
                    response_payload={"item_name": item_name},
                )

            return HandlerResult(
                next_state=ConversationState.CONFIRMING_ITEM,
                response_key="confirm_item",
                response_payload={"item_name": confirmation.get("value_name")},
            )

        return HandlerResult(
            next_state=ConversationState.ERROR_RECOVERY,
            response_key="confirmation_state_error",
        )

    def _resolve_candidate_item_from_confirmation(
        self,
        *,
        context: ConversationContext,
        user_text: str,
    ) -> _CandidateMatch | None:
        confirmation = context.awaiting_confirmation_for or {}
        candidate_item_ids = confirmation.get("candidate_item_ids") or []
        candidate_item_names = confirmation.get("candidate_item_names") or []

        if not candidate_item_ids or not candidate_item_names:
            return None

        utterance = self._normalize_candidate_utterance(user_text)
        if not utterance:
            return None

        candidates: list[tuple[str, str]] = []
        for item_id, item_name in zip(candidate_item_ids, candidate_item_names):
            if not item_id or not item_name:
                continue
            candidates.append((str(item_id), str(item_name)))

        # 1) exact normalized full-name match only
        for item_id, item_name in candidates:
            normalized_name = normalize_text(item_name)
            if utterance == normalized_name:
                try:
                    return _CandidateMatch(
                        item=self.menu_repo.get_item(item_id),
                        match_type="exact",
                    )
                except KeyError:
                    return None

        # 2) strong token-set match only (strict)
        utterance_tokens = self._token_set(utterance)
        if not utterance_tokens:
            return None

        best_item: MenuItem | None = None
        best_score = 0.0

        for item_id, item_name in candidates:
            normalized_name = normalize_text(item_name)
            name_tokens = self._token_set(normalized_name)
            if not name_tokens:
                continue

            overlap = len(utterance_tokens & name_tokens)
            score = overlap / max(len(name_tokens), 1)

            # Require strong match:
            # - at least 80% of candidate tokens covered
            # - utterance must have at least 2 tokens
            # This prevents "add chicken" => "Chicken Taco"
            if score >= 0.80 and len(utterance_tokens) >= 2 and score > best_score:
                try:
                    best_item = self.menu_repo.get_item(item_id)
                    best_score = score
                except KeyError:
                    continue

        if best_item is not None:
            return _CandidateMatch(item=best_item, match_type="strong")

        return None

    def _normalize_candidate_utterance(self, text: str) -> str:
        normalized = normalize_text(text)
        if not normalized:
            return ""

        for prefix in FILLER_PREFIXES:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):].strip()
                break

        return normalized

    def _token_set(self, text: str) -> set[str]:
        return {token for token in text.split() if token}

    def _enter_add_flow_for_item(
        self,
        *,
        context: ConversationContext,
        item: MenuItem,
    ) -> HandlerResult:
        existing_quantity = context.quantity

        context.reset_task()
        context.pending_action = PendingAction.ADD_ITEM
        context.quantity = existing_quantity

        context.current_item_id = item.item_id
        context.current_item_name = item.name
        context.candidate_item_id = item.item_id
        context.awaiting_confirmation_for = None
        context.awaiting_flow_confirmation = False
        context.interrupt_proposal = None
        context.pending_add_item = build_pending_add_item(item)

        step = determine_next_add_item_step(context)

        if step.next_state == ConversationState.FINALIZING_ADD_ITEM:
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="item_added_successfully",
                response_payload={
                    "item_name": item.name,
                    "quantity": context.quantity or 1,
                },
                command=build_add_item_command(context),
                reset_context=True,
            )

        return HandlerResult(
            next_state=step.next_state,
            response_key=step.response_key,
            response_payload=step.response_payload,
        )

    def _repeat_response_key(self, confirmation: dict) -> str:
        reason = confirmation.get("reason")
        if reason == "category_detected":
            return "confirm_item_from_category"
        return "confirm_item_ambiguous"

    def _first_candidate_name(self, confirmation: dict) -> str | None:
        names = confirmation.get("candidate_item_names") or []
        for name in names:
            if isinstance(name, str) and name.strip():
                return name.strip()
        return None

    def _move_to_selected_candidate_state(
            self,
            *,
            context: ConversationContext,
            item: MenuItem,
            previous_confirmation: dict,
    ) -> HandlerResult:
        context.candidate_item_id = item.item_id
        context.current_item_name = item.name
        context.awaiting_confirmation_for = {
            "type": "item",
            "reason": "candidate_selected",
            "value_id": item.item_id,
            "value_name": item.name,
            "previous_confirmation": dict(previous_confirmation),
        }

        return HandlerResult(
            next_state=ConversationState.CONFIRMING_ITEM,
            response_key="confirm_item",
            response_payload={"item_name": item.name},
        )

    def _resolve_fresh_item_attempt(
            self,
            *,
            context: ConversationContext,
            user_text: str,
    ) -> HandlerResult | None:
        normalized_text = normalize_text(user_text)
        if not normalized_text:
            return None

        slots = getattr(context, "last_slots", None) or []
        item_slot_value = None
        for slot in slots:
            if str(getattr(slot, "name", "")).upper() in {"ITEM", "MENU_ITEM"}:
                value = getattr(slot, "value", None)
                if isinstance(value, str) and value.strip():
                    item_slot_value = value.strip()
                    break

        if item_slot_value:
            result = self.menu_repo.resolve_menu_query_from_slots(
                user_text="",
                slots=slots,
                fallback_to_text=False,
                limit=5,
            )
            requested_text = item_slot_value
        else:
            result = self.menu_repo.resolve_menu_query(
                normalized_text,
                limit=5,
            )
            requested_text = normalized_text

        if result.type == MenuQueryType.ITEM and result.item is not None:
            return self._move_to_selected_candidate_state(
                context=context,
                item=result.item,
                previous_confirmation=context.awaiting_confirmation_for or {},
            )

        if (
                result.type == MenuQueryType.CATEGORY_SINGLE_ITEM
                and result.items
                and len(result.items) == 1
        ):
            return self._move_to_selected_candidate_state(
                context=context,
                item=result.items[0],
                previous_confirmation=context.awaiting_confirmation_for or {},
            )

        if result.type in {MenuQueryType.ITEM_AMBIGUOUS, MenuQueryType.CATEGORY_AMBIGUOUS}:
            payload = {
                "reason": "multiple_matches",
                "query": requested_text,
            }

            if result.matched_items:
                payload["candidate_item_ids"] = [item.item_id for item in result.matched_items]
                payload["candidate_item_names"] = [item.name for item in result.matched_items]

            if result.matched_categories:
                payload["candidate_category_names"] = [
                    category.get("name")
                    for category in result.matched_categories
                    if category.get("name")
                ]

            context.awaiting_confirmation_for = {
                "type": "item",
                "reason": "multiple_matches",
                "query": requested_text,
                "candidate_item_ids": payload.get("candidate_item_ids", []),
                "candidate_item_names": payload.get("candidate_item_names", []),
                "candidate_category_names": payload.get("candidate_category_names", []),
            }

            context.candidate_item_id = None
            context.current_item_name = None

            return HandlerResult(
                next_state=ConversationState.CONFIRMING_ITEM,
                response_key="confirm_item_ambiguous",
                response_payload=payload,
            )

        if result.type == MenuQueryType.CATEGORY:
            payload = {
                "reason": "category_detected",
                "query": requested_text,
                "category_id": result.category_id,
                "category_name": result.category_name,
                "candidate_item_ids": [item.item_id for item in (result.items or [])],
                "candidate_item_names": [item.name for item in (result.items or [])],
            }

            context.awaiting_confirmation_for = {
                "type": "item",
                "reason": "category_detected",
                "query": requested_text,
                "category_id": result.category_id,
                "category_name": result.category_name,
                "candidate_item_ids": payload["candidate_item_ids"],
                "candidate_item_names": payload["candidate_item_names"],
            }

            context.candidate_item_id = None
            context.current_item_name = None

            return HandlerResult(
                next_state=ConversationState.CONFIRMING_ITEM,
                response_key="confirm_item_from_category",
                response_payload=payload,
            )

        return None