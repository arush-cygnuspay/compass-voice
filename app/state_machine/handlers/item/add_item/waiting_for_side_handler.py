# app/state_machine/handlers/item/waiting_for_side_handler.py
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

        idx = context.current_side_group_index

        # -------------------------
        # End of side groups
        # -------------------------
        if idx >= len(item.side_groups):
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

        group = item.side_groups[idx]

        # -------------------------
        # Options request
        # -------------------------
        if self._wants_options(user_text):
            top_choices = [c.name for c in get_top_k_choices(group.choices, k=3)]
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_SIDE,
                response_key="list_side_options",
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
        # Global exit on DENY (no / nothing else / done / that's it)
        # -------------------------
        if intent == Intent.DENY:
            if group.is_required:
                top_choices = [c.name for c in get_top_k_choices(group.choices, k=3)]
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_SIDE,
                    response_key="required_side_cannot_skip",
                    response_payload={"top_choices": top_choices, "group_name": group.name},
                )
            context.current_side_group_index = len(item.side_groups)
            return self._advance_after_side(context, item)

        # -------------------------
        # Skip OPTIONAL side (silence)
        # -------------------------
        if (not user_text.strip()) and not group.is_required:
            context.current_side_group_index += 1
            return self._advance_after_side(context, item)

        # -------------------------
        # Match side choice (supports multiple mentions)
        # -------------------------
        matched_choices = []
        for chunk in self._split_candidates(user_text):
            choice = match_choice(chunk, group.choices)
            if choice and choice not in matched_choices:
                matched_choices.append(choice)

        if not matched_choices:
            top_choices = [c.name for c in get_top_k_choices(group.choices, k=3)]
            if not user_text.strip():
                # Required group needs guidance
                if group.is_required:
                    return HandlerResult(
                        next_state=ConversationState.WAITING_FOR_SIDE,
                        response_key="required_side_cannot_skip",
                        response_payload={"top_choices": top_choices, "group_name": group.name},
                    )
            if len(user_text.split()) == 1 and len(user_text.strip()) <= 2:
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_SIDE,
                    response_key="clarify_side_choice",
                    response_payload={
                        "top_choices": top_choices,
                        "group_name": group.name,
                    },
                )
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_SIDE,
                response_key="repeat_side_options",
                response_payload={
                    "top_choices": top_choices,
                    "group_name": group.name,
                },
            )

        if group.max_selector == 1 and len(matched_choices) > 1:
            top_choices = [c.name for c in get_top_k_choices(group.choices, k=3)]
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_SIDE,
                response_key="too_many_side_choices",
                response_payload={
                    "top_choices": top_choices,
                    "group_name": group.name,
                },
            )

        # -------------------------
        # Commit side
        # -------------------------
        chosen = matched_choices[0]
        context.selected_side_groups.setdefault(group.group_id, []).append(
            chosen.item_id
        )

        context.current_side_group_index += 1

        return self._advance_after_side(context, item)

    def _advance_after_side(self, context: ConversationContext, item):
        if context.current_side_group_index < len(item.side_groups):
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_SIDE,
                response_key="ask_for_side",
            )

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
        return bool(re.search(r"\b(no|nope|nopes|none|nothing|skip|done|that's it|thats it|nothing else)\b", text, flags=re.IGNORECASE))
