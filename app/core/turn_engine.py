# app/core/turn_engine.py
from dataclasses import dataclass
from typing import Optional

from app.cart.read_models.cart_summary_builder import CartSummaryBuilder
from app.nlu.intent_resolution.intent_resolver import resolve_intent
from app.nlu.query_normalization.pipeline import QueryNormalizationPipeline
from app.session.session import Session
from app.state_machine.handlers.item.add_item.adding_item_handler import AddItemHandler
from app.state_machine.handlers.item.add_item.waiting_for_modifier_handler import WaitingForModifierHandler
from app.state_machine.handlers.item.add_item.waiting_for_side_handler import WaitingForSideHandler
from app.state_machine.handlers.item.add_item.waiting_for_size_handler import WaitingForSizeHandler
from app.state_machine.handlers.item.confirming_handler import ConfirmingHandler
from app.state_machine.handlers.order.confirm_order_handler import ConfirmOrderHandler
from app.state_machine.handlers.order.start_order_handler import StartOrderHandler
from app.state_machine.handlers.payment.waiting_for_payment_handler import WaitingForPaymentHandler
from app.state_machine.state_router import StateRouter
from app.menu.repository import MenuRepository


# Internal DTO for Turn Result
@dataclass
class TurnOutput:
    response_key: str
    response_payload: Optional[dict] = None


class TurnEngine:
    """
    Stateless turn processor.
    Session is the single source of truth.
    """

    def __init__(self, router: StateRouter, menu_repo: MenuRepository):
        self.router = router
        self.menu_repo = menu_repo
        self.cart_summary_builder = CartSummaryBuilder(menu_repo)

        self.normalizer = QueryNormalizationPipeline()

        # Explicit handler registry
        self.handlers = {
            "add_item_handler": AddItemHandler(
                menu_repo=menu_repo,
            ),
            "waiting_for_side_handler": WaitingForSideHandler(menu_repo),
            "confirming_handler": ConfirmingHandler(menu_repo),
            "waiting_for_modifier_handler": WaitingForModifierHandler(menu_repo),
            "waiting_for_size_handler": WaitingForSizeHandler(menu_repo),
            "start_order_handler": StartOrderHandler(self.cart_summary_builder),
            "confirming_order_handler": ConfirmOrderHandler(),
            "waiting_for_payment_handler": WaitingForPaymentHandler(),
        }

    def process_turn(
            self,
            session: Session,
            user_text: str,
    ) -> TurnOutput:

        # 1️⃣ Pure NLU
        intent_result = resolve_intent(
            user_text,
            state=session.conversation_state
        )

        # 🔥 NORMALIZE ONCE, HERE
        normalized_text = self.normalizer.normalize(
            text=user_text,
            intent=intent_result.intent,
            state=session.conversation_state,
        )

        # 2️⃣ Route based on STATE + intent
        route = self.router.route(
            state=session.conversation_state,
            intent_result=intent_result,
        )

        if not route.allowed:
            return TurnOutput(
                response_key="intent_not_allowed"
            )

        handler = self.handlers.get(route.handler_name)
        if not handler:
            return TurnOutput(
                response_key="handler_not_implemented"
            )

        # 3️⃣ Execute handler
        result = handler.handle(
            intent=intent_result.intent,  # ← pass original intent
            context=session.conversation_context,
            user_text=normalized_text,
            session=session,
        )

        # 🔑 Apply side-effects centrally
        if result.command:
            self._apply_command(session, result.command)

        if result.reset_context:
            session.conversation_context.reset()

        session.conversation_state = result.next_state
        session.last_intent = intent_result.intent
        session.last_response_key = result.response_key
        session.turn_count += 1

        return TurnOutput(
            response_key=result.response_key,
            response_payload=result.response_payload,
        )

    def _apply_command(self, session: Session, command: dict) -> None:
        command_type = command["type"]

        if command_type == "ADD_ITEM_TO_CART":
            from app.cart.cart_item import CartItem

            payload = command["payload"]

            cart_item = CartItem.create(
                item_id=payload["item_id"],
                quantity=payload["quantity"],
                variant_id=payload.get("variant_id"),
                sides=payload.get("sides", {}),
                modifiers=payload.get("modifiers", {}),
            )

            session.cart.add_item(cart_item)

        else:
            raise ValueError(f"Unknown command type: {command_type}")


