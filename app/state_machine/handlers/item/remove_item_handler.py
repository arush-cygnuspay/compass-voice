# app/state_machine/handlers/item/remove_item_handler.py
from app.session.session import Session
from app.state_machine.base_handler import BaseHandler
from app.state_machine.handler_result import HandlerResult
from app.state_machine.conversation_state import ConversationState
from app.state_machine.context import ConversationContext
from app.nlu.intent_resolution.intent import Intent
from app.menu.repository import MenuRepository
from app.utils.item_matching import score_item


class RemoveItemHandler(BaseHandler):
    """
    Handles initial item removal intent.
    Resolves item from cart using matching techniques.
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

        # Defensive: this should never happen post-refiner
        if intent != Intent.REMOVE_ITEM:
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="unhandled_intent",
            )

        # Check if cart is empty
        if session.cart.is_empty():
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="cart_is_empty",
            )

        # -----------------------------
        # Resolve item from cart
        # -----------------------------
        matched_cart_item = self._match_cart_item(user_text, session)

        if not matched_cart_item:
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="item_not_found_in_cart",
            )

        # -----------------------------
        # Initialize remove-item context
        # -----------------------------
        context.pending_action = "remove"
        context.candidate_item_id = matched_cart_item.cart_item_id
        context.current_item_id = matched_cart_item.item_id
        
        # Get item name for confirmation message
        menu_item = self.menu_repo.get_item(matched_cart_item.item_id)
        context.current_item_name = menu_item.name

        # Set up confirmation
        context.awaiting_confirmation_for = {
            "type": "remove_item",
            "cart_item_id": matched_cart_item.cart_item_id,
            "item_id": matched_cart_item.item_id,
            "item_name": menu_item.name,
        }

        return HandlerResult(
            next_state=ConversationState.REMOVING_ITEM,
            response_key="confirm_remove_item",
            response_payload={
                "item_name": menu_item.name,
                "quantity": matched_cart_item.quantity,
            },
        )

    def _match_cart_item(self, user_text: str, session: Session):
        """
        Match user text to a cart item using the same matching techniques
        as item and choice matching.
        """
        cart_items = session.cart.get_items()
        
        if not cart_items:
            return None

        best_cart_item = None
        best_score = 0.0

        for cart_item in cart_items:
            # Get menu item to access name
            menu_item = self.menu_repo.get_item(cart_item.item_id)
            
            # Use the same scoring function as item matching
            score = score_item(user_text, menu_item.name)
            
            if score > best_score:
                best_score = score
                best_cart_item = cart_item

        # Confidence threshold (same as item resolution)
        return best_cart_item if best_score >= 2.5 else None
