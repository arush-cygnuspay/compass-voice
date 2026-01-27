# app/state_machine/handlers/item/waiting_for_size_handler.py
from dataclasses import dataclass
from app.session.session import Session
from app.state_machine.base_handler import BaseHandler
from app.state_machine.handler_result import HandlerResult
from app.state_machine.conversation_state import ConversationState
from app.state_machine.context import ConversationContext
from app.nlu.intent_resolution.intent import Intent
from app.menu.repository import MenuRepository
from app.utils.choice_matching import match_choice


@dataclass
class _VariantChoice:
    variant_id: str
    name: str


class WaitingForSizeHandler(BaseHandler):
    """
    Resolves size for the current size_target (item).

    Size selection is REQUIRED:
    - DENY is not allowed
    - Unknown input re-prompts with options
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

        if intent == Intent.CANCEL:
            context.reset()
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="action_cancelled",
            )

        item = self.menu_repo.get_item(context.current_item_id)
        if not item or item.pricing.mode != "variant":
            return HandlerResult(
                next_state=ConversationState.ERROR_RECOVERY,
                response_key="item_context_missing",
            )

        if intent == Intent.DENY:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_SIZE,
                response_key="required_size_cannot_skip",
            )

        variant_choices = [
            _VariantChoice(v.variant_id, v.label)
            for v in item.pricing.variants
        ]

        matched = match_choice(user_text, variant_choices)
        if not matched:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_SIZE,
                response_key="repeat_size_options",
            )

        context.selected_variant_id = matched.variant_id
        context.size_target = None

        # ---------- Continue structure ----------
        if item.side_groups:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_SIDE,
                response_key="ask_for_side",
            )

        if item.modifier_groups:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_MODIFIER,
                response_key="ask_for_modifier",
            )

        return HandlerResult(
            next_state=ConversationState.WAITING_FOR_QUANTITY,
            response_key="ask_for_quantity",
            response_payload={"item_name": item.name},
        )

