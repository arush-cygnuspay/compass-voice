# app/core/turn_engine.py
from dataclasses import dataclass
from typing import Optional

from app.cart.read_models.cart_summary_builder import CartSummaryBuilder
from app.nlu.intent_refinement.intent_refiner import IntentRefiner
from app.nlu.intent_resolution.intent import Intent
from app.nlu.intent_resolution.intent_resolver import resolve_intent
from app.nlu.intent_resolution.intent_result import IntentResult
from app.nlu.query_normalization.base import basic_cleanup
from app.nlu.query_normalization.noise_cleaner import clean_stt_noise
from app.nlu.query_normalization.pipeline import QueryNormalizationPipeline
from app.session.session import Session
from app.state_machine.conversation_state import ConversationState
from app.state_machine.handler_result import HandlerResult
from app.state_machine.handlers.cart.cart_handlers import CartHandler, ShowingCartHandler, ShowingTotalHandler
from app.state_machine.handlers.common.cancellation_confirmation_handler import CancellationConfirmationHandler
from app.state_machine.handlers.info.ask_menu_info_handler import AskMenuInfoHandler
from app.state_machine.handlers.info.ask_price_handler import AskPriceHandler
from app.state_machine.handlers.item.add_item.adding_item_handler import AddItemHandler
from app.state_machine.handlers.item.add_item.waiting_for_modifier_handler import WaitingForModifierHandler
from app.state_machine.handlers.common.waiting_for_quantity_handler import WaitingForQuantityHandler
from app.state_machine.handlers.item.add_item.waiting_for_side_handler import WaitingForSideHandler
from app.state_machine.handlers.item.add_item.waiting_for_size_handler import WaitingForSizeHandler
from app.state_machine.handlers.item.confirming_handler import ConfirmingHandler
from app.state_machine.handlers.item.remove_item_handler import RemoveItemHandler
from app.state_machine.handlers.item.removing_item_handler import RemovingItemHandler
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
        self.intent_refiner = IntentRefiner(menu_repo)

        # Explicit handler registry
        self.handlers = {
            "add_item_handler": AddItemHandler(
                menu_repo=menu_repo,
            ),
            "waiting_for_side_handler": WaitingForSideHandler(menu_repo),
            "confirming_handler": ConfirmingHandler(menu_repo),
            "waiting_for_modifier_handler": WaitingForModifierHandler(menu_repo),
            "waiting_for_size_handler": WaitingForSizeHandler(menu_repo),
            "remove_item_handler": RemoveItemHandler(menu_repo),
            "removing_item_handler": RemovingItemHandler(),
            "start_order_handler": StartOrderHandler(self.cart_summary_builder),
            "confirming_order_handler": ConfirmOrderHandler(),
            "waiting_for_payment_handler": WaitingForPaymentHandler(),

            # Cart utilities
            "cart_handler": CartHandler(self.cart_summary_builder),
            "showing_cart_handler": ShowingCartHandler(),
            "showing_total_handler": ShowingTotalHandler(),

            "cancellation_confirmation_handler": CancellationConfirmationHandler(),

            # Info Handlers
            "ask_menu_info_handler": AskMenuInfoHandler(menu_repo),
            "ask_price_handler": AskPriceHandler(menu_repo),

            "waiting_for_quantity_handler": WaitingForQuantityHandler(),
        }

    def process_turn(self, session: Session, user_text: str) -> TurnOutput:

        preclean_text = basic_cleanup(user_text)
        stt_cleaned_text = clean_stt_noise(preclean_text)

        # ---------------- NLU ----------------
        intent_result = resolve_intent(stt_cleaned_text, state=session.conversation_state)

        normalized_text = self.normalizer.normalize(
            text=stt_cleaned_text,
            intent=intent_result.intent,
            state=session.conversation_state,
        )

        refined_intent = self.intent_refiner.refine(
            intent=intent_result.intent,
            normalized_text=normalized_text,
            state=session.conversation_state,
        )

        intent_result = IntentResult(
            intent=refined_intent,
            raw_text=intent_result.raw_text,
        )

        # ---------------- ROUTER ----------------
        route = self.router.route(
            state=session.conversation_state,
            intent_result=intent_result,
        )

        if not route.allowed:
            return TurnOutput(response_key="intent_not_allowed")

        handler = self.handlers.get(route.handler_name)

        result = handler.handle(
            intent=intent_result.intent,
            context=session.conversation_context,
            user_text=normalized_text,
            session=session,
        )

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

        elif command_type == "CLEAR_CART":
            session.cart.clear()

        elif command_type == "REMOVE_ITEM_FROM_CART":
            payload = command["payload"]
            cart_item_id = payload["cart_item_id"]
            removed = session.cart.remove_item(cart_item_id)
            if not removed:
                # This should rarely happen as validation happens in handler
                # But log it for debugging
                pass

        else:
            raise ValueError(f"Unknown command type: {command_type}")

    def _apply_result(self, session: Session, result: HandlerResult) -> None:
        if result.command:
            self._apply_command(session, result.command)

        if result.reset_context:
            session.conversation_context.reset()

        session.conversation_state = result.next_state
        session.last_response_key = result.response_key
        session.turn_count += 1

