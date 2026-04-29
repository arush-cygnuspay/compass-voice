"""Handler registry + dispatch + reprompt guardrail.

Owns the handlers dict (constructed internally from upstream
dependencies), the post-dispatch reprompt guardrail, and the command
delegation. Behavior moved verbatim from ``turn_engine.py``.
"""
from __future__ import annotations

from typing import Any

from app.cart.read_models.cart_summary_builder import CartSummaryBuilder
from app.contracts.command_result import CommandResult
from app.core.command_executor import CommandExecutor
from app.core.response_builder import ResponseBuilder
from app.core.turn_diagnostics import REPROMPT_RESPONSE_KEYS, TurnDiagnostics
from app.menu.repository import MenuRepository
from app.nlu.intent_resolution.intent import Intent
from app.services.checkout_service import CheckoutService
from app.services.sms_service import SmsService
from app.session.session import Session
from app.state_machine.control_intent_resolver import log_control_intent_event
from app.state_machine.handler_result import HandlerResult
from app.state_machine.handlers.cart.cart_handlers import CartHandler
from app.state_machine.handlers.common.cancellation_confirmation_handler import (
    CancellationConfirmationHandler,
)
from app.state_machine.handlers.common.waiting_for_quantity_handler import (
    WaitingForQuantityHandler,
)
from app.state_machine.handlers.delivery.waiting_for_delivery_address_collection_handler import (
    WaitingForDeliveryAddressCollectionHandler,
)
from app.state_machine.handlers.delivery.waiting_for_delivery_eligibility_handler import (
    WaitingForDeliveryEligibilityHandler,
)
from app.state_machine.handlers.info.ask_menu_info_handler import AskMenuInfoHandler
from app.state_machine.handlers.info.ask_price_handler import AskPriceHandler
from app.state_machine.handlers.item.add_item.add_item_handler import AddItemHandler
from app.state_machine.handlers.item.add_item.waiting_for_modifier_handler import (
    WaitingForModifierHandler,
)
from app.state_machine.handlers.item.add_item.waiting_for_side_handler import (
    WaitingForSideHandler,
)
from app.state_machine.handlers.item.add_item.waiting_for_side_size_handler import (
    WaitingForSideSizeHandler,
)
from app.state_machine.handlers.item.add_item.waiting_for_size_handler import (
    WaitingForSizeHandler,
)
from app.state_machine.handlers.item.confirming_handler import ConfirmingHandler
from app.state_machine.handlers.item.modifying_item_handler import ModifyingItemHandler
from app.state_machine.handlers.item.remove_item_handler import RemoveItemHandler
from app.state_machine.handlers.item.removing_item_handler import RemovingItemHandler
from app.state_machine.handlers.order.confirm_order_handler import ConfirmOrderHandler
from app.state_machine.handlers.order.start_order_handler import StartOrderHandler
from app.state_machine.handlers.order.waiting_for_order_type_handler import (
    WaitingForOrderTypeHandler,
)
from app.state_machine.handlers.payment.waiting_for_checkout_completion_handler import (
    WaitingForCheckoutCompletionHandler,
)
from app.state_machine.handlers.payment.waiting_for_payment_handler import (
    WaitingForPaymentHandler,
)
from app.state_machine.handlers.system.waiting_for_caller_device_type_handler import (
    WaitingForCallerDeviceTypeHandler,
)
from app.state_machine.models.conversation_state import ConversationState


class HandlerDispatcher:
    """Builds and owns the handlers dict; handles dispatch, reprompt
    guardrail, and command application."""

    def __init__(
        self,
        *,
        menu_repo: MenuRepository,
        cart_summary_builder: CartSummaryBuilder,
        sms_service: SmsService,
        checkout_service: CheckoutService,
        responder: ResponseBuilder,
        command_executor: CommandExecutor,
        diagnostics: TurnDiagnostics,
    ) -> None:
        self.menu_repo = menu_repo
        self.cart_summary_builder = cart_summary_builder
        self.sms_service = sms_service
        self.checkout_service = checkout_service
        self.responder = responder
        self.command_executor = command_executor
        self.diagnostics = diagnostics

        self.handlers: dict[str, Any] = {
            "add_item_handler": AddItemHandler(menu_repo=menu_repo),
            "waiting_for_side_handler": WaitingForSideHandler(menu_repo),
            "waiting_for_modifier_handler": WaitingForModifierHandler(menu_repo),
            "waiting_for_size_handler": WaitingForSizeHandler(menu_repo),
            "waiting_for_side_size_handler": WaitingForSideSizeHandler(menu_repo),
            "waiting_for_quantity_handler": WaitingForQuantityHandler(),
            "confirming_handler": ConfirmingHandler(menu_repo),
            "modifying_item_handler": ModifyingItemHandler(menu_repo),
            "remove_item_handler": RemoveItemHandler(menu_repo),
            "removing_item_handler": RemovingItemHandler(),
            "start_order_handler": StartOrderHandler(cart_summary_builder),
            "confirming_order_handler": ConfirmOrderHandler(
                cart_summary_builder,
                sms_service,
                checkout_service,
            ),
            "waiting_for_payment_handler": WaitingForPaymentHandler(
                cart_summary_builder,
                checkout_service,
            ),
            "waiting_for_checkout_completion_handler": WaitingForCheckoutCompletionHandler(
                checkout_service,
            ),
            "cart_handler": CartHandler(cart_summary_builder),
            "cancellation_confirmation_handler": CancellationConfirmationHandler(),
            "ask_menu_info_handler": AskMenuInfoHandler(menu_repo),
            "ask_price_handler": AskPriceHandler(menu_repo),
            "waiting_for_caller_device_type_handler": WaitingForCallerDeviceTypeHandler(),
            "waiting_for_order_type_handler": WaitingForOrderTypeHandler(),
            "waiting_for_delivery_eligibility_handler": WaitingForDeliveryEligibilityHandler(),
            "waiting_for_delivery_address_collection_handler": WaitingForDeliveryAddressCollectionHandler(
                cart_summary_builder,
                checkout_service,
            ),
        }

    def get_handler(self, name: str) -> Any:
        return self.handlers.get(name)

    def _apply_command(
        self,
        session: Session,
        command: dict[str, Any],
    ) -> CommandResult:
        return self.command_executor.execute(session, command)

    def _apply_reprompt_guardrail(
        self,
        *,
        session: Session,
        state_before: ConversationState,
        result: HandlerResult,
    ) -> HandlerResult:
        response_key = result.response_key or ""
        is_quantity_reprompt = (
            state_before == ConversationState.WAITING_FOR_QUANTITY
            and response_key == "ask_for_quantity"
        )
        if response_key not in REPROMPT_RESPONSE_KEYS and not is_quantity_reprompt:
            return result

        field = self.diagnostics._infer_prompt_field_for_response(
            response_key=response_key,
            session=session,
        )
        if not field:
            return result

        current_count = int(session.reprompt_count_by_field.get(field, 0) or 0)
        next_count = current_count + 1

        # Always stamp the 1-based miss count so renderers can tier their prompts.
        payload = dict(result.response_payload or {})
        payload["reprompt_count"] = next_count
        result.response_payload = payload

        if next_count < 3:
            return result

        payload["reprompt_escalation"] = True
        original_response_key = result.response_key
        if field == "side":
            result.response_key = "list_side_options"
        elif field == "modifier":
            result.response_key = "list_modifier_options"
        elif field == "size":
            result.response_key = "repeat_size_options"
        elif field == "side_size":
            result.response_key = "repeat_side_size_options"
        elif field == "quantity":
            result.response_key = "invalid_quantity_option"
        log_control_intent_event(
            "reprompt_escalated",
            field=field,
            attempts=next_count,
            original_response_key=original_response_key,
            escalated_response_key=result.response_key,
            state=(
                session.conversation_state.value
                if session.conversation_state
                else None
            ),
        )
        return result
