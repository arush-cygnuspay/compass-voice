# app/state_machine/handlers/item/add_item/waiting_for_modifier_handler.py
from app.session.session import Session
from app.state_machine.base_handler import BaseHandler
from app.state_machine.handler_result import HandlerResult
from app.state_machine.conversation_state import ConversationState
from app.state_machine.context import ConversationContext
from app.nlu.intent_resolution.intent import Intent
from app.menu.repository import MenuRepository
from app.utils.choice_matching import match_choice
from app.utils.top_k_choices import get_top_k_choices
import re


class WaitingForModifierHandler(BaseHandler):
    """
    Handles modifier selection for the current item.
    Processes one modifier group at a time.
    """

    def __init__(self, menu_repo: MenuRepository) -> None:
        self.menu_repo = menu_repo

    def handle(
        self,
        intent: Intent,
        context: ConversationContext,
        user_text: str,
        session: Session = None,
    ) -> HandlerResult:

        # -------------------------
        # Global cancel
        # -------------------------
        if intent == Intent.CANCEL:
            context.reset()
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="action_cancelled",
            )

        item = self.menu_repo.get_item(context.current_item_id)
        if not item:
            return HandlerResult(
                next_state=ConversationState.ERROR_RECOVERY,
                response_key="item_context_missing",
            )

        idx = context.current_modifier_group_index
        total = len(item.modifier_groups)

        # -------------------------
        # No modifier groups left
        # -------------------------
        if idx >= total:
            return self._finalize_item(context, item)

        group = item.modifier_groups[idx]

        # -------------------------
        # Options request
        # -------------------------
        if self._wants_options(user_text):
            top_choices = [c.name for c in get_top_k_choices(group.choices, k=3)]
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_MODIFIER,
                response_key="list_modifier_options",
                response_payload={
                    "top_choices": top_choices,
                    "group_name": group.name,
                },
            )

        # -------------------------
        # Treat negative phrases as deny intent
        # -------------------------
        if self._is_negative(user_text):
            intent = Intent.DENY

        # -------------------------
        # Skip current modifier group on DENY (no/none/that's all) for optional
        # -------------------------
        if intent == Intent.DENY:
            if group.is_required:
                top_choices = [c.name for c in get_top_k_choices(group.choices, k=3)]
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_MODIFIER,
                    response_key="required_modifier_cannot_skip",
                    response_payload={
                        "top_choices": top_choices,
                        "group_name": group.name,
                    },
                )
            context.current_modifier_group_index = total
            return self._finalize_item(context, item)

        # -------------------------
        # Open-capture modifier choices (can be multiple)
        # -------------------------
        matched_ids: list[str] = []
        invalid_terms: list[str] = []

        for chunk in self._split_candidates(user_text):
            choice = match_choice(chunk, group.choices)
            if choice:
                if choice.modifier_id not in matched_ids:
                    matched_ids.append(choice.modifier_id)
            else:
                invalid_terms.append(chunk.strip())

        if matched_ids:
            if group.max_selector == 1 and len(matched_ids) > 1:
                top_choices = [c.name for c in get_top_k_choices(group.choices, k=3)]
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_MODIFIER,
                    response_key="too_many_modifier_choices",
                    response_payload={
                        "top_choices": top_choices,
                        "group_name": group.name,
                    },
                )
            context.selected_modifier_groups.setdefault(group.group_id, []).extend(
                matched_ids
            )

        if invalid_terms:
            top_choices = [c.name for c in get_top_k_choices(group.choices, k=3)]
            if len(user_text.split()) == 1 and len(user_text.strip()) <= 2:
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_MODIFIER,
                    response_key="clarify_modifier_choice",
                    response_payload={
                        "top_choices": top_choices,
                        "group_name": group.name,
                    },
                )
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_MODIFIER,
                response_key="repeat_modifier_options",
                response_payload={
                    "top_choices": top_choices,
                    "invalid_terms": invalid_terms,
                    "group_name": group.name,
                },
            )

        if group.is_required and not matched_ids:
            top_choices = [c.name for c in get_top_k_choices(group.choices, k=3)]
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_MODIFIER,
                response_key="required_modifier_cannot_skip",
                response_payload={
                    "top_choices": top_choices,
                    "group_name": group.name,
                },
            )

        # Advance after processing current group
        context.current_modifier_group_index += 1

        if context.current_modifier_group_index >= total:
            return self._finalize_item(context, item)

        return HandlerResult(
            next_state=ConversationState.WAITING_FOR_MODIFIER,
            response_key="ask_for_modifier",
        )

    def _finalize_item(self, context: ConversationContext, item) -> HandlerResult:
        return HandlerResult(
            next_state=ConversationState.IDLE,
            response_key="item_added_successfully",
            command={
                "type": "ADD_ITEM_TO_CART",
                "payload": {
                    "item_id": context.current_item_id,
                    "quantity": context.quantity or 1,
                    "variant_id": context.selected_variant_id,
                    "sides": context.selected_side_groups,
                    "modifiers": context.selected_modifier_groups,
                },
            },
            response_payload={
                "item_id": context.current_item_id,
                "item_name": item.name,
                "quantity": context.quantity or 1,
            },
            reset_context=True,
        )

    def _split_candidates(self, text: str) -> list[str]:
        if not text:
            return []
        parts = re.split(r",| and | & |\+", text, flags=re.IGNORECASE)
        return [p.strip() for p in parts if p.strip()]

    def _wants_options(self, text: str) -> bool:
        if not text:
            return False
        return bool(re.search(r"\b(option|options|choices?)\b", text, flags=re.IGNORECASE))

    def _is_negative(self, text: str) -> bool:
        if not text:
            return False
        return bool(re.search(r"\b(no|nope|nopes|none|nothing|no modifications|no mods|skip|done|that's it|thats it|nothing else)\b", text, flags=re.IGNORECASE))
