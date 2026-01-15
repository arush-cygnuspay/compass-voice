# app/state_machine/handlers/item/confirming_handler.py

from app.state_machine.base_handler import BaseHandler
from app.state_machine.handler_result import HandlerResult
from app.state_machine.conversation_state import ConversationState
from app.state_machine.context import ConversationContext
from app.nlu.intent import Intent
from app.menu.menu_repository import MenuRepository


class ConfirmingHandler(BaseHandler):
    """
    Handles confirmation steps for item, side, modifier, size, quantity.
    All mutations happen ONLY on confirmation here.
    """

    def __init__(self, menu_repo: MenuRepository) -> None:
        self.menu_repo = menu_repo

    def handle(
        self,
        intent: Intent,
        context: ConversationContext,
    ) -> HandlerResult:

        confirmation = context.awaiting_confirmation_for

        if not confirmation:
            return HandlerResult(
                next_state=ConversationState.ERROR_RECOVERY,
                response_key="confirmation_state_error",
            )

        # Global cancellation
        if intent == Intent.CANCEL:
            context.reset()
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="action_cancelled",
            )

        confirmation_type = confirmation.get("type")

        # -------------------------------------------------
        # ITEM CONFIRMATION
        # -------------------------------------------------
        if confirmation_type == "item":

            if intent == Intent.DENY:
                context.candidate_item_id = None
                context.awaiting_confirmation_for = None
                return HandlerResult(
                    next_state=ConversationState.IDLE,
                    response_key="item_confirmation_denied",
                )

            if intent == Intent.CONFIRM:
                item_id = context.candidate_item_id
                context.current_item_id = item_id
                context.candidate_item_id = None
                context.awaiting_confirmation_for = None

                item = self.menu_repo.items.get(item_id)

                if not item:
                    return HandlerResult(
                        next_state=ConversationState.ERROR_RECOVERY,
                        response_key="item_context_missing",
                    )

                if item.side_groups:
                    return HandlerResult(
                        next_state=ConversationState.WAITING_FOR_SIDE,
                        response_key="ask_for_side",
                    )

                if item.pricing.mode == "variant":
                    return HandlerResult(
                        next_state=ConversationState.WAITING_FOR_SIZE,
                        response_key="ask_for_size",
                    )

                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_QUANTITY,
                    response_key="ask_for_quantity",
                )

        # -------------------------------------------------
        # SIDE CONFIRMATION
        # -------------------------------------------------
        if confirmation_type == "side":

            group_id = confirmation["group_id"]
            value_id = confirmation["value_id"]

            if intent == Intent.DENY:
                context.awaiting_confirmation_for = None
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_SIDE,
                    response_key="ask_for_side",
                )

            if intent == Intent.CONFIRM:
                item = self.menu_repo.items.get(context.current_item_id)
                if not item:
                    return HandlerResult(
                        next_state=ConversationState.ERROR_RECOVERY,
                        response_key="item_context_missing",
                    )

                # Find side group
                group = next(
                    (g for g in item.side_groups if g.group_id == group_id),
                    None,
                )

                if not group:
                    return HandlerResult(
                        next_state=ConversationState.ERROR_RECOVERY,
                        response_key="side_group_missing",
                    )

                selected = context.selected_side_groups.setdefault(group_id, [])

                # Enforce max selector
                if len(selected) >= group.max_selector:
                    context.awaiting_confirmation_for = None
                    return HandlerResult(
                        next_state=ConversationState.WAITING_FOR_SIDE,
                        response_key="side_selection_limit_reached",
                    )

                selected.append(value_id)
                context.awaiting_confirmation_for = None

                # Check if more sides are needed
                for g in item.side_groups:
                    current = context.selected_side_groups.get(g.group_id, [])
                    if g.is_required and len(current) < g.min_selector:
                        return HandlerResult(
                            next_state=ConversationState.WAITING_FOR_SIDE,
                            response_key="ask_for_side_group",
                        )

                # Move forward
                if item.pricing.mode == "variant":
                    return HandlerResult(
                        next_state=ConversationState.WAITING_FOR_SIZE,
                        response_key="ask_for_size",
                    )

                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_QUANTITY,
                    response_key="ask_for_quantity",
                )

        # -------------------------------------------------
        # MODIFIER CONFIRMATION
        # -------------------------------------------------
        if confirmation_type == "modifier":

            group_id = confirmation["group_id"]
            value_id = confirmation["value_id"]

            if intent == Intent.DENY:
                context.awaiting_confirmation_for = None
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_MODIFIER,
                    response_key="ask_for_modifier",
                )

            if intent == Intent.CONFIRM:
                item = self.menu_repo.items.get(context.current_item_id)
                if not item:
                    return HandlerResult(
                        next_state=ConversationState.ERROR_RECOVERY,
                        response_key="item_context_missing",
                    )

                group = next(
                    (g for g in item.modifier_groups if g.group_id == group_id),
                    None,
                )

                if not group:
                    return HandlerResult(
                        next_state=ConversationState.ERROR_RECOVERY,
                        response_key="modifier_group_missing",
                    )

                selected = context.selected_modifier_groups.setdefault(group_id, [])

                if len(selected) >= group.max_selector:
                    context.awaiting_confirmation_for = None
                    return HandlerResult(
                        next_state=ConversationState.WAITING_FOR_MODIFIER,
                        response_key="modifier_limit_reached",
                    )

                selected.append(value_id)
                context.awaiting_confirmation_for = None

                # Continue modifier flow
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_MODIFIER,
                    response_key="ask_for_modifier",
                )

        # -------------------------------------------------
        # FALLBACK: repeat confirmation
        # -------------------------------------------------
        return HandlerResult(
            next_state=ConversationState.CONFIRMING,
            response_key="repeat_confirmation",
        )
