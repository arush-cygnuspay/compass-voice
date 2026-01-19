# app/state_machine/handlers/item/add_item/waiting_for_modifier_handler.py
from app.session.session import Session
from app.state_machine.base_handler import BaseHandler
from app.state_machine.handler_result import HandlerResult
from app.state_machine.conversation_state import ConversationState
from app.state_machine.context import ConversationContext
from app.nlu.intent_resolution.intent import Intent
from app.menu.repository import MenuRepository
from app.utils.choice_matching import match_choice


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

        # Global cancel
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

        # -------------------------------------------------
        # NO MODIFIER GROUPS LEFT → FINALIZE ITEM
        # -------------------------------------------------
        if idx >= total:
            return self._finalize_item(context, item)

        group = item.modifier_groups[idx]

        # -------------------------------------------------
        # USER SKIPS OPTIONAL MODIFIER
        # -------------------------------------------------
        if intent == Intent.DENY:
            next_index = idx + 1
            context.current_modifier_group_index = next_index

            if next_index >= total:
                return self._finalize_item(context, item)

            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_MODIFIER,
                response_key="ask_for_modifier",
            )

        # -------------------------------------------------
        # MATCH MODIFIER
        # -------------------------------------------------
        matched_choice = match_choice(user_text, group.choices)

        if not matched_choice:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_MODIFIER,
                response_key="repeat_modifier_options",
            )

        # -------------------------------------------------
        # COMMIT MODIFIER
        # -------------------------------------------------
        context.selected_modifier_groups.setdefault(group.group_id, []).append(
            matched_choice.modifier_id
        )

        next_index = idx + 1
        context.current_modifier_group_index = next_index

        if next_index >= total:
            return self._finalize_item(context, item)

        return HandlerResult(
            next_state=ConversationState.WAITING_FOR_MODIFIER,
            response_key="ask_for_modifier",
        )

    # -------------------------------------------------
    # FINALIZATION (single authority)
    # -------------------------------------------------
    def _finalize_item(
        self,
        context: ConversationContext,
        item,
    ) -> HandlerResult:
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

