# app/state_machine/handlers/item/waiting_for_side_handler.py

from app.state_machine.base_handler import BaseHandler
from app.state_machine.handler_result import HandlerResult
from app.state_machine.conversation_state import ConversationState
from app.state_machine.context import ConversationContext
from app.nlu.intent_resolution.intent import Intent
from app.menu.repository import MenuRepository


class WaitingForSideHandler(BaseHandler):
    """
    Handles side selection for items that require sides.
    Enforces selection one side group at a time.
    """

    def __init__(self, menu_repo: MenuRepository) -> None:
        self.menu_repo = menu_repo

    def handle(
        self,
        intent: Intent,
        context: ConversationContext,
        user_text: str,
    ) -> HandlerResult:

        # Cancellation is always allowed
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

        # Are we out of side groups?
        if context.current_side_group_index >= len(item.side_groups):
            # After sides → modifiers (if any) → size → quantity
            if item.modifier_groups:
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_MODIFIER,
                    response_key="ask_for_modifier",
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

        group = item.side_groups[context.current_side_group_index]

        # -----------------------------
        # Matching helpers
        # -----------------------------
        def normalize(text: str) -> str:
            return text.lower().strip()

        def tokens(text: str) -> set[str]:
            return set(normalize(text).split())

        user_norm = normalize(user_text)
        user_tokens = tokens(user_text)

        matched_choice = None

        for choice in group.choices:
            choice_norm = normalize(choice.name)
            choice_tokens = tokens(choice.name)

            # 1️⃣ Phrase containment (BEST)
            if choice_norm in user_norm:
                matched_choice = choice
                break

            # 2️⃣ All tokens present (order independent)
            if choice_tokens.issubset(user_tokens):
                matched_choice = choice
                break

            # 3️⃣ Partial token overlap (fallback)
            if user_tokens & choice_tokens:
                matched_choice = choice

        # ❌ No match at all
        if not matched_choice:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_SIDE,
                response_key="repeat_side_options",
            )

        # -----------------------------
        # Commit selection
        # -----------------------------
        context.selected_side_groups.setdefault(group.group_id, []).append(
            matched_choice.item_id
        )

        context.current_side_group_index += 1

        # Ask next side group
        if context.current_side_group_index < len(item.side_groups):
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_SIDE,
                response_key="ask_for_side",
            )

        # Continue flow
        if item.modifier_groups:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_MODIFIER,
                response_key="ask_for_modifier",
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

