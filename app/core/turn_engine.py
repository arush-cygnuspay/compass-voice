# app/core/turn_engine.py
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from app.cart.read_models.cart_summary_builder import CartSummaryBuilder
from app.core.flow_control.flow_decision import FlowAction
from app.core.flow_control.flow_control_policy import FlowControlPolicy
from app.core.response_builder import ResponseBuilder
from app.logging.nlu_csv_logger import NluCsvLogger
from app.menu.query_result import MenuQueryType
from app.menu.repository import MenuRepository
from app.ml.intent.inference_intent import IntentBundle
from app.ml.slot.inference_slot import SlotBundle
from app.nlu.intent_resolution.intent import Intent
from app.nlu.intent_resolution.intent_result import IntentResult
from app.nlu.nlu_resolver import resolve_nlu
from app.nlu.query_normalization.text_preprocessor import preprocess_turn_text
from app.services.checkout_service import CheckoutService
from app.services.sms_service import SmsSendRequest, SmsSendResult, SmsService
from app.session.session import Session
from app.state_machine.handlers.delivery.waiting_for_delivery_address_collection_handler import (
    WaitingForDeliveryAddressCollectionHandler,
)
from app.state_machine.handlers.delivery.waiting_for_delivery_eligibility_handler import (
    WaitingForDeliveryEligibilityHandler,
)
from app.state_machine.handlers.payment.waiting_for_checkout_completion_handler import (
    WaitingForCheckoutCompletionHandler,
)
from app.state_machine.models.conversation_state import ConversationState
from app.state_machine.handler_result import HandlerResult
from app.state_machine.handlers.cart.cart_handlers import CartHandler
from app.state_machine.handlers.common.cancellation_confirmation_handler import (
    CancellationConfirmationHandler,
)
from app.state_machine.handlers.order.waiting_for_order_type_handler import WaitingForOrderTypeHandler
from app.state_machine.handlers.system.waiting_for_caller_device_type_handler import (
    HUMAN_AGENT_TRANSFER_NUMBER,
    WaitingForCallerDeviceTypeHandler,
)
from app.state_machine.handlers.common.waiting_for_quantity_handler import (
    WaitingForQuantityHandler,
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
from app.state_machine.handlers.payment.payment_flow_support import verify_payment_for_order
from app.state_machine.handlers.payment.waiting_for_payment_handler import (
    WaitingForPaymentHandler,
)
from app.state_machine.resume_prompt_builder import ResumePromptBuilder
from app.state_machine.state_router import StateRouter


@dataclass(frozen=True, slots=True)
class TurnOutput:
    response_key: str
    response_payload: dict[str, Any] | None = None
    internal_response_text: str | None = None
    spoken_response_text: str | None = None
    end_call_after_playback: bool = False
    # If set, the voice transport layer should redirect the live call to
    # this PSTN number after the spoken response finishes playing. Used
    # to hand landline callers off to a human agent.
    transfer_call_to_number: str | None = None


INTENT_MIN_CONF = float(os.getenv("COMPASS_INTENT_CONF_THRESHOLD", "0.55"))
TURN_TIMING_ENABLED = os.getenv("COMPASS_TURN_TIMING_ENABLED", "0") == "1"
ROUTE_DEBUG_ENABLED = os.getenv("COMPASS_ROUTE_DEBUG_ENABLED", "0") == "1"

CONFIRMING_ORDER_EXIT_TO_IDLE_INTENTS: set[Intent] = {
    Intent.ADD_ITEM,
    Intent.REMOVE_ITEM,
    Intent.REPLACE_ITEM,
    Intent.MODIFY_ITEM,
    Intent.UNDO_LAST,
    Intent.ASK_ITEM_INFO,
    Intent.ASK_MENU_INFO,
    Intent.ASK_OPTIONS,
    Intent.AVAILABILITY_QUERY,
    Intent.BROWSE_MENU,
    Intent.BROWSE_CATEGORY,
    Intent.RECOMMENDATION_QUERY,
    Intent.SHOW_MENU,
}

WAITING_STATE_ALLOWED_CONTROL_INTENTS = {
    Intent.DENY,
    Intent.CANCEL,
    Intent.CANCEL_ORDER,
    Intent.ASK_OPTIONS,
    Intent.SHOW_CART,
    Intent.SHOW_TOTAL,
    Intent.ASK_PRICE,
    Intent.ASK_ITEM_INFO,
    Intent.ASK_MENU_INFO,
    Intent.AVAILABILITY_QUERY,
    Intent.BROWSE_MENU,
    Intent.BROWSE_CATEGORY,
    Intent.RECOMMENDATION_QUERY,
    Intent.SHOW_MENU,
    # allow true task-switch requests to reach the waiting handlers
    Intent.ADD_ITEM,
    Intent.REMOVE_ITEM,
    Intent.MODIFY_ITEM,
}

DELIVERY_GATING_ALLOWED_CONTROL_INTENTS = {
    Intent.AFFIRM,
    Intent.CONFIRM,
    Intent.DENY,
    Intent.CANCEL,
    Intent.CANCEL_ORDER,
    # Ordering intents pass through so handlers can redirect gracefully
    # instead of treating food names as delivery area / zip input.
    Intent.ADD_ITEM,
    Intent.REMOVE_ITEM,
    Intent.MODIFY_ITEM,
    Intent.SHOW_MENU,
    Intent.ASK_MENU_INFO,
    Intent.ASK_PRICE,
    Intent.SHOW_CART,
    Intent.SHOW_TOTAL,
}


class TurnEngine:
    def __init__(
        self,
        router: StateRouter,
        menu_repo: MenuRepository,
        intent_bundle: IntentBundle,
        slot_bundle: SlotBundle,
        responder: ResponseBuilder,
        sms_service: SmsService,
        nlu_logger: NluCsvLogger | None = None,
    ) -> None:
        self.router = router
        self.menu_repo = menu_repo
        self.intent_bundle = intent_bundle
        self.slot_bundle = slot_bundle
        self.cart_summary_builder = CartSummaryBuilder(menu_repo)
        self.flow_policy = FlowControlPolicy()
        self.nlu_logger = nlu_logger or NluCsvLogger()
        self.resume_prompt_builder = ResumePromptBuilder()
        self.responder = responder
        self.sms_service = sms_service
        self.checkout_service = CheckoutService()

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
            "start_order_handler": StartOrderHandler(self.cart_summary_builder),
            "confirming_order_handler": ConfirmOrderHandler(
                self.cart_summary_builder,
                self.sms_service,
                self.checkout_service,
            ),
            "waiting_for_payment_handler": WaitingForPaymentHandler(
                self.cart_summary_builder,
                self.checkout_service,
            ),
            "waiting_for_checkout_completion_handler": WaitingForCheckoutCompletionHandler(
                self.checkout_service,
            ),
            "cart_handler": CartHandler(self.cart_summary_builder),
            "cancellation_confirmation_handler": CancellationConfirmationHandler(),
            "ask_menu_info_handler": AskMenuInfoHandler(menu_repo),
            "ask_price_handler": AskPriceHandler(menu_repo),
            "waiting_for_caller_device_type_handler": WaitingForCallerDeviceTypeHandler(),
            "waiting_for_order_type_handler": WaitingForOrderTypeHandler(),
            "waiting_for_delivery_eligibility_handler": WaitingForDeliveryEligibilityHandler(),
            "waiting_for_delivery_address_collection_handler": WaitingForDeliveryAddressCollectionHandler(
                self.cart_summary_builder,
                self.checkout_service,
            ),
        }

    def process_turn(
        self,
        session: Session,
        user_text: str,
        trace: Any | None = None,
    ) -> TurnOutput:
        t_total_start = time.perf_counter()
        ctx = session.conversation_context

        if session.conversation_state == ConversationState.COMPLETED:
            return self._hydrate_output(
                session=session,
                output=TurnOutput(
                    response_key="order_completed",
                    response_payload=session.last_response_payload,
                    end_call_after_playback=True,
                ),
            )

        # Once a landline caller has been handed off to a human agent the
        # voice transport layer will tear down the WebSocket / call. If a
        # spurious turn still arrives in this state we just acknowledge
        # and tell the bridge to end the agent's side of the call.
        if session.conversation_state == ConversationState.TRANSFERRING_TO_HUMAN_AGENT:
            return self._hydrate_output(
                session=session,
                output=TurnOutput(
                    response_key="transferring_to_human_agent",
                    response_payload={
                        "transfer_number": HUMAN_AGENT_TRANSFER_NUMBER,
                    },
                    transfer_call_to_number=HUMAN_AGENT_TRANSFER_NUMBER,
                    end_call_after_playback=True,
                ),
            )

        # Caller-device-type gate. Must run BEFORE the order-type gate
        # because landline callers are routed to a live human and never
        # reach the pickup / delivery question.
        if session.conversation_state == ConversationState.WAITING_FOR_CALLER_DEVICE_TYPE:
            device_handler: WaitingForCallerDeviceTypeHandler = self.handlers[
                "waiting_for_caller_device_type_handler"
            ]
            device_result = device_handler.handle(
                intent=Intent.UNKNOWN,
                context=ctx,
                user_text=user_text,
                session=session,
            )

            session.conversation_state = device_result.next_state
            self._apply_session_response(
                session=session,
                intent=Intent.UNKNOWN,
                response_key=device_result.response_key,
                response_payload=device_result.response_payload,
            )

            transfer_number: str | None = None
            command = device_result.command or {}
            if command.get("type") == "transfer_call":
                transfer_number = command.get("transfer_number")

            return self._hydrate_output(
                session=session,
                output=TurnOutput(
                    response_key=device_result.response_key,
                    response_payload=device_result.response_payload,
                    transfer_call_to_number=transfer_number,
                    # Once we hand the call off there is nothing for the
                    # voice agent to do, so end the agent leg cleanly
                    # after the goodbye plays. The transfer command is
                    # carried out by the transport layer and supersedes
                    # a plain hangup if both are set.
                    end_call_after_playback=transfer_number is not None,
                ),
            )

        # Auto payment-check probe injected by the transport layer after
        # PAYMENT_AUTO_CHECK_DELAY_SECONDS of silence.  Bypass NLU entirely
        # and call the payment verifier directly — O(1) path, no model inference.
        if user_text == "__auto_payment_check__":
            return self._handle_auto_payment_check(session)

        self._normalize_order_type_gate_state(session)

        if session.conversation_state == ConversationState.WAITING_FOR_ORDER_TYPE:
            handler: WaitingForOrderTypeHandler = self.handlers["waiting_for_order_type_handler"]
            gate_result = handler.handle(
                intent=Intent.UNKNOWN,
                context=ctx,
                user_text=user_text,
                session=session,
            )

            session.conversation_state = gate_result.next_state
            self._apply_session_response(
                session=session,
                intent=Intent.UNKNOWN,
                response_key=gate_result.response_key,
                response_payload=gate_result.response_payload,
            )

            return self._hydrate_output(
                session=session,
                output=TurnOutput(
                    response_key=gate_result.response_key,
                    response_payload=gate_result.response_payload,
                ),
            )

        state_before = session.conversation_state
        ctx.last_user_text = user_text

        self._trace_set_initial_fields(
            trace=trace,
            session=session,
            user_text=user_text,
            state_before=state_before,
            engine_start_monotonic=t_total_start,
        )

        t0 = time.perf_counter()
        preprocessed = preprocess_turn_text(user_text)
        cleaned_text = preprocessed.cleaned_text
        normalized_text = preprocessed.normalized_text
        t_preprocess = time.perf_counter() - t0

        self._trace_set_attr(trace, "cleaned_text", cleaned_text)
        self._trace_set_attr(trace, "normalized_text", normalized_text)

        t0 = time.perf_counter()
        nlu = resolve_nlu(
            raw_text=cleaned_text,
            normalized_text=normalized_text,
            state=session.conversation_state,
            pending_action=ctx.pending_action,
            intent_bundle=self.intent_bundle,
            slot_bundle=self.slot_bundle,
        )
        ctx.set_last_nlu(user_text=cleaned_text, nlu=nlu)
        t_nlu = time.perf_counter() - t0

        self._trace_set_nlu_fields(trace=trace, nlu=nlu)

        detected_intent = (
            nlu.effective_intent
            if nlu.intent_confidence >= INTENT_MIN_CONF
            else Intent.UNKNOWN
        )
        intent_result = IntentResult(
            intent=detected_intent,
            raw_text=nlu.normalized_text,
        )

        delivery_gating_states = {
            ConversationState.WAITING_FOR_DELIVERY_ELIGIBILITY,
            ConversationState.WAITING_FOR_DELIVERY_ADDRESS_COLLECTION,
        }

        generic_waiting_states = {
            ConversationState.WAITING_FOR_SIDE,
            ConversationState.WAITING_FOR_SIDE_SIZE,
            ConversationState.WAITING_FOR_MODIFIER,
            ConversationState.WAITING_FOR_SIZE,
            ConversationState.WAITING_FOR_QUANTITY,
        }

        if session.conversation_state in delivery_gating_states:
            allowed_control_intents = DELIVERY_GATING_ALLOWED_CONTROL_INTENTS
        elif session.conversation_state in generic_waiting_states:
            allowed_control_intents = WAITING_STATE_ALLOWED_CONTROL_INTENTS
        else:
            allowed_control_intents = set()

        if (
                session.conversation_state in delivery_gating_states | generic_waiting_states
                and intent_result.intent not in allowed_control_intents
        ):
            intent_result = IntentResult(
                intent=Intent.UNKNOWN,
                raw_text=nlu.normalized_text,
            )

        session.conversation_state = self._rewrite_confirming_order_to_idle_if_needed(
            session=session,
            intent=intent_result.intent,
        )

        intent_result, shortcut_output = self._apply_idle_shortcuts(session, intent_result)
        if shortcut_output is not None:
            self._apply_session_response(
                session=session,
                intent=intent_result.intent,
                response_key=shortcut_output.response_key,
                response_payload=shortcut_output.response_payload,
            )

            self._log_if_enabled(
                session=session,
                state_before=state_before,
                nlu=nlu,
                result=None,
                response_key=shortcut_output.response_key,
            )

            total_ms = (time.perf_counter() - t_total_start) * 1000.0
            self._finalize_trace_and_timing(
                trace=trace,
                session=session,
                response_key=shortcut_output.response_key,
                total_start_monotonic=t_total_start,
                total_ms=total_ms,
                preprocess_ms=t_preprocess * 1000.0,
                nlu_ms=t_nlu * 1000.0,
                flow_ms=0.0,
                route_ms=0.0,
                handler_ms=0.0,
            )
            return self._hydrate_output(
                session=session,
                output=shortcut_output,
            )

        intent_result = self._rewrite_idle_unknown_menu_followup(
            session=session,
            intent_result=intent_result,
            normalized_text=nlu.normalized_text,
        )

        t0 = time.perf_counter()
        flow = self.flow_policy.evaluate(
            state=session.conversation_state,
            intent=intent_result.intent,
            context=ctx,
        )
        t_flow = time.perf_counter() - t0

        if flow.action == FlowAction.BLOCK:
            payload = dict(flow.response_payload or {})
            response_key = flow.response_key or "flow_blocked"

            self._apply_session_response(
                session=session,
                intent=intent_result.intent,
                response_key=response_key,
                response_payload=payload,
            )

            self._log_if_enabled(
                session=session,
                state_before=state_before,
                nlu=nlu,
                result=None,
                response_key=response_key,
            )

            total_ms = (time.perf_counter() - t_total_start) * 1000.0
            self._finalize_trace_and_timing(
                trace=trace,
                session=session,
                response_key=response_key,
                total_start_monotonic=t_total_start,
                total_ms=total_ms,
                preprocess_ms=t_preprocess * 1000.0,
                nlu_ms=t_nlu * 1000.0,
                flow_ms=t_flow * 1000.0,
                route_ms=0.0,
                handler_ms=0.0,
            )
            return self._hydrate_output(
                session=session,
                output=TurnOutput(
                    response_key=response_key,
                    response_payload=payload,
                ),
            )

        if flow.action == FlowAction.CANCEL:
            ctx.awaiting_flow_confirmation = True
            ctx.return_state = session.conversation_state
            ctx.interrupt_proposal = None

            session.conversation_state = ConversationState.CANCELLATION_CONFIRMATION
            response_key = flow.response_key or "flow_guard_confirm_cancel"
            response_payload = dict(flow.response_payload or {})

            self._apply_session_response(
                session=session,
                intent=intent_result.intent,
                response_key=response_key,
                response_payload=response_payload,
            )

            self._log_if_enabled(
                session=session,
                state_before=state_before,
                nlu=nlu,
                result=None,
                response_key=response_key,
            )

            total_ms = (time.perf_counter() - t_total_start) * 1000.0
            self._finalize_trace_and_timing(
                trace=trace,
                session=session,
                response_key=response_key,
                total_start_monotonic=t_total_start,
                total_ms=total_ms,
                preprocess_ms=t_preprocess * 1000.0,
                nlu_ms=t_nlu * 1000.0,
                flow_ms=t_flow * 1000.0,
                route_ms=0.0,
                handler_ms=0.0,
            )
            return self._hydrate_output(
                session=session,
                output=TurnOutput(
                    response_key=response_key,
                    response_payload=response_payload,
                ),
            )

        if flow.action == FlowAction.HANDLE_READONLY_INTERRUPT:
            readonly_output = self._handle_readonly_interrupt(
                session=session,
                state_before=state_before,
                intent_result=intent_result,
                nlu=nlu,
                trace=trace,
            )
            if readonly_output is not None:
                total_ms = (time.perf_counter() - t_total_start) * 1000.0
                self._finalize_trace_and_timing(
                    trace=trace,
                    session=session,
                    response_key=readonly_output.response_key,
                    total_start_monotonic=t_total_start,
                    total_ms=total_ms,
                    preprocess_ms=t_preprocess * 1000.0,
                    nlu_ms=t_nlu * 1000.0,
                    flow_ms=t_flow * 1000.0,
                    route_ms=0.0,
                    handler_ms=0.0,
                )
                return readonly_output

        if flow.action == FlowAction.REWRITE and flow.effective_intent is not None:
            intent_result = IntentResult(
                intent=flow.effective_intent,
                raw_text=intent_result.raw_text,
            )

        t0 = time.perf_counter()
        route = self.router.route(session.conversation_state, intent_result)
        t_route = time.perf_counter() - t0

        if ROUTE_DEBUG_ENABLED:
            print(
                "[ROUTE]",
                {
                    "state": session.conversation_state.value,
                    "intent": intent_result.intent.value,
                    "handler": route.handler_name,
                    "allowed": route.allowed,
                },
            )

        if not route.allowed or not route.handler_name:
            payload = {
                "state": session.conversation_state.value,
                "intent": intent_result.intent.value,
            }

            self._apply_session_response(
                session=session,
                intent=intent_result.intent,
                response_key="intent_not_allowed",
                response_payload=payload,
            )

            self._log_if_enabled(
                session=session,
                state_before=state_before,
                nlu=nlu,
                result=None,
                response_key="intent_not_allowed",
            )

            total_ms = (time.perf_counter() - t_total_start) * 1000.0
            self._finalize_trace_and_timing(
                trace=trace,
                session=session,
                response_key="intent_not_allowed",
                total_start_monotonic=t_total_start,
                total_ms=total_ms,
                preprocess_ms=t_preprocess * 1000.0,
                nlu_ms=t_nlu * 1000.0,
                flow_ms=t_flow * 1000.0,
                route_ms=t_route * 1000.0,
                handler_ms=0.0,
            )
            return self._hydrate_output(
                session=session,
                output=TurnOutput(
                    response_key="intent_not_allowed",
                    response_payload=payload,
                ),
            )

        handler = self.handlers.get(route.handler_name)
        if handler is None:
            raise KeyError(f"Handler not registered: {route.handler_name}")

        t0 = time.perf_counter()
        result: HandlerResult = handler.handle(
            intent=intent_result.intent,
            context=ctx,
            user_text=nlu.normalized_text,
            session=session,
        )
        t_handler = time.perf_counter() - t0

        command_result: dict[str, Any] | None = None
        transfer_number: str | None = None

        if result.command:
            command_type = result.command.get("type")
            if command_type == "transfer_call":
                transfer_number = result.command.get("transfer_number")
                command_result = {
                    "ok": True,
                    "transport_only": True,
                    "transfer_number": transfer_number,
                }
            else:
                command_result = self._apply_command(session, result.command)

            print(
                "[COMMAND RESULT]",
                {
                    "command": result.command,
                    "result": command_result,
                },
            )

            if not command_result.get("ok", False):
                command_type = result.command.get("type")
                command_payload = result.command.get("payload") or {}

                if command_type == "SEND_SMS":
                    template = command_payload.get("template")
                    delivery = session.conversation_context.delivery_address
                    attempts_made = int(command_result.get("attempts_made", 1) or 1)

                    if template == "checkout_link":
                        # _apply_command already retried internally.
                        # If it still failed after those attempts, fall back to voice immediately.
                        if attempts_made >= 2:
                            delivery.source = "voice"
                            session.conversation_context.current_prompt_field = "delivery_seed_confirmation"
                            result = HandlerResult(
                                next_state=ConversationState.WAITING_FOR_DELIVERY_ADDRESS_COLLECTION,
                                response_key="checkout_link_failed_fallback_voice",
                                response_payload={
                                    "area": delivery.area,
                                    "postal_code": delivery.postal_code,
                                    "order_number": delivery.order_number,
                                    "error_code": command_result.get("error_code"),
                                    "error_message": command_result.get("error_message"),
                                },
                            )
                        else:
                            result = HandlerResult(
                                next_state=ConversationState.CONFIRMING_ORDER,
                                response_key="checkout_link_send_failed",
                                response_payload={
                                    "order_number": delivery.order_number,
                                    "error_code": command_result.get("error_code"),
                                    "error_message": command_result.get("error_message"),
                                },
                            )

                    elif template == "payment_link":
                        # Payment-link flow also retried internally.
                        # If still failed, apologize and stop progression.
                        if attempts_made >= 2:
                            result = HandlerResult(
                                next_state=session.conversation_state,
                                response_key="payment_link_unavailable_now",
                                response_payload={
                                    "order_number": delivery.order_number,
                                    "error_code": command_result.get("error_code"),
                                    "error_message": command_result.get("error_message"),
                                },
                            )
                        else:
                            result = HandlerResult(
                                next_state=session.conversation_state,
                                response_key="payment_link_send_failed",
                                response_payload={
                                    "order_number": delivery.order_number,
                                    "error_code": command_result.get("error_code"),
                                    "error_message": command_result.get("error_message"),
                                },
                            )

        if result.reset_context:
            ctx.reset()

        session.conversation_state = result.next_state

        # ── Multi-item queue drain ────────────────────────────
        # When an item was just added to cart and we return to IDLE,
        # check if there are queued items from a multi-item utterance.
        # If so, auto-start the next queued item.
        queue_drain_result = self._try_drain_item_queue(
            session=session,
            current_result=result,
        )
        if queue_drain_result is not None:
            result = queue_drain_result

        self._apply_session_response(
            session=session,
            intent=intent_result.intent,
            response_key=result.response_key,
            response_payload=result.response_payload,
        )

        self._log_if_enabled(
            session=session,
            state_before=state_before,
            nlu=nlu,
            result=result,
            response_key=result.response_key,
        )

        total_ms = (time.perf_counter() - t_total_start) * 1000.0
        self._finalize_trace_and_timing(
            trace=trace,
            session=session,
            response_key=result.response_key,
            total_start_monotonic=t_total_start,
            total_ms=total_ms,
            preprocess_ms=t_preprocess * 1000.0,
            nlu_ms=t_nlu * 1000.0,
            flow_ms=t_flow * 1000.0,
            route_ms=t_route * 1000.0,
            handler_ms=t_handler * 1000.0,
            command=result.command,
        )

        return self._hydrate_output(
            session=session,
            output=TurnOutput(
                response_key=result.response_key,
                response_payload=result.response_payload,
                internal_response_text=getattr(result, "internal_response_text", None),
                spoken_response_text=getattr(result, "spoken_response_text", None),
                end_call_after_playback=(
                    result.next_state == ConversationState.COMPLETED
                    or transfer_number is not None
                ),
                transfer_call_to_number=transfer_number,
            ),
        )

    def _try_drain_item_queue(
        self,
        *,
        session: Session,
        current_result: HandlerResult,
    ) -> HandlerResult | None:
        """
        After an item is added to cart, check if there are queued items
        from a multi-item utterance. If so, dequeue the next item and
        start its add-item flow.

        Returns a new HandlerResult if a queued item was started,
        or None if no queue drain is needed.
        """
        ctx = session.conversation_context

        # Only drain when item was just successfully added and we'd go to IDLE
        if current_result.response_key != "item_added_successfully":
            return None
        if session.conversation_state != ConversationState.IDLE:
            return None
        if not ctx.pending_item_queue:
            return None

        # Pop the next item from the queue
        next_item = ctx.pending_item_queue.popleft()
        remaining_count = len(ctx.pending_item_queue)

        # Build the previous item's added summary
        prev_payload = current_result.response_payload or {}
        prev_item_name = prev_payload.get("item_name", "item")
        prev_quantity = prev_payload.get("quantity", 1)

        # Feed the queued item through AddItemHandler
        add_handler: Any = self.handlers.get("add_item_handler")
        if add_handler is None:
            return None

        # Set up context for the new item
        ctx.reset_task()
        ctx.pending_action = None

        # Provide the queued item's quantity
        if next_item.quantity and next_item.quantity > 0:
            ctx.quantity = next_item.quantity

        # Inject the preserved segment slots from multi-item parsing so
        # that modifier/side/size prefilling has full NLU context.
        # Falls back to a synthetic ITEM slot when no segment slots exist.
        if next_item.segment_slots:
            ctx.last_slots = next_item.segment_slots
        elif next_item.item_slot_value:
            from app.nlu.nlu_result import SlotValue as SlotValueClass
            ctx.last_slots = (
                SlotValueClass(
                    name="ITEM",
                    value=next_item.item_slot_value,
                    raw=next_item.item_slot_value,
                    start=None,
                    end=None,
                    confidence=1.0,
                ),
            )

        # Run the handler with the queued item's text
        from app.nlu.intent_resolution.intent import Intent as IntentEnum
        next_result: HandlerResult = add_handler.handle(
            intent=IntentEnum.ADD_ITEM,
            context=ctx,
            user_text=next_item.raw_text,
            session=session,
        )

        # Apply any command from the next item (e.g., instant add)
        if next_result.command:
            self._apply_command(session, next_result.command)

        if next_result.reset_context:
            ctx.reset()

        session.conversation_state = next_result.next_state

        # Build a combined response: "Added X. Now for Y..."
        next_payload = dict(next_result.response_payload or {})
        next_payload["queue_transition"] = True
        next_payload["prev_item_name"] = prev_item_name
        next_payload["prev_quantity"] = prev_quantity
        next_payload["next_item_name"] = next_item.item_slot_value or next_item.raw_text
        next_payload["remaining_queue_count"] = remaining_count

        # If the next item was also instantly added (all slots prefilled),
        # recursively drain the queue
        if next_result.response_key == "item_added_successfully" and ctx.pending_item_queue:
            deeper = self._try_drain_item_queue(
                session=session,
                current_result=next_result,
            )
            if deeper is not None:
                # Chain: "Added X. Added Y. Now for Z..."
                deeper_payload = dict(deeper.response_payload or {})
                deeper_payload["chain_prev_items"] = next_payload.get("chain_prev_items", []) + [
                    {"name": prev_item_name, "quantity": prev_quantity}
                ]
                return HandlerResult(
                    next_state=deeper.next_state,
                    response_key=deeper.response_key,
                    response_payload=deeper_payload,
                    command=deeper.command,
                    reset_context=False,
                )

        return HandlerResult(
            next_state=next_result.next_state,
            response_key=next_result.response_key,
            response_payload=next_payload,
            command=None,  # command already applied above
            reset_context=False,
        )

    def _apply_session_response(
        self,
        *,
        session: Session,
        intent: Intent,
        response_key: str,
        response_payload: dict[str, Any] | None,
    ) -> None:
        session.last_intent = intent
        session.last_response_key = response_key
        session.last_response_payload = response_payload
        session.turn_count += 1

    def _log_if_enabled(
        self,
        *,
        session: Session,
        state_before: ConversationState,
        nlu: Any,
        result: HandlerResult | None,
        response_key: str,
    ) -> None:
        if not self.nlu_logger.enabled:
            return

        context_snapshot = self._snapshot_context_for_logging(session)
        self._log_nlu_and_turn(
            session=session,
            state_before=state_before,
            context_snapshot=context_snapshot,
            nlu=nlu,
            result=result,
            response_key=response_key,
        )

    def _finalize_trace_and_timing(
        self,
        *,
        trace: Any | None,
        session: Session,
        response_key: str,
        total_start_monotonic: float,
        total_ms: float,
        preprocess_ms: float,
        nlu_ms: float,
        flow_ms: float,
        route_ms: float,
        handler_ms: float,
        command: dict[str, Any] | None = None,
    ) -> None:
        self._trace_finalize(
            trace=trace,
            session=session,
            response_key=response_key,
            total_start_monotonic=total_start_monotonic,
            total_ms=total_ms,
            preprocess_ms=preprocess_ms,
            nlu_ms=nlu_ms,
            flow_ms=flow_ms,
            route_ms=route_ms,
            handler_ms=handler_ms,
            command=command,
        )

        self._maybe_print_timing(
            total_ms=total_ms,
            preprocess_ms=preprocess_ms,
            nlu_ms=nlu_ms,
            flow_ms=flow_ms,
            route_ms=route_ms,
            handler_ms=handler_ms,
        )

    def _maybe_print_timing(
        self,
        *,
        total_ms: float,
        preprocess_ms: float,
        nlu_ms: float,
        flow_ms: float,
        route_ms: float,
        handler_ms: float,
    ) -> None:
        if not TURN_TIMING_ENABLED:
            return

        print(
            "[TURN_TIMING]",
            {
                "total_ms": round(total_ms, 3),
                "preprocess_ms": round(preprocess_ms, 3),
                "nlu_ms": round(nlu_ms, 3),
                "flow_ms": round(flow_ms, 3),
                "route_ms": round(route_ms, 3),
                "handler_ms": round(handler_ms, 3),
            },
        )

    def _handle_readonly_interrupt(
        self,
        *,
        session: Session,
        state_before: ConversationState,
        intent_result: IntentResult,
        nlu: Any,
        trace: Any | None = None,
    ) -> TurnOutput | None:
        handler_name = self._readonly_interrupt_handler_name(intent_result.intent)
        if handler_name is None:
            return None

        handler = self.handlers.get(handler_name)
        if handler is None:
            raise KeyError(f"Handler not registered: {handler_name}")

        preserved_state = session.conversation_state
        context_snapshot = (
            self._snapshot_context_for_logging(session)
            if self.nlu_logger.enabled
            else {}
        )

        result: HandlerResult = handler.handle(
            intent=intent_result.intent,
            context=session.conversation_context,
            user_text=nlu.normalized_text,
            session=session,
        )

        if result.command is not None:
            raise ValueError(
                f"Read-only interrupt handler {handler_name} returned command={result.command}. "
                "Read-only interrupt handlers must not mutate session state."
            )

        if result.reset_context:
            raise ValueError(
                f"Read-only interrupt handler {handler_name} attempted reset_context=True."
            )

        resume = self.resume_prompt_builder.build(session)

        session.conversation_state = preserved_state
        session.last_intent = intent_result.intent

        if resume is None:
            session.last_response_key = result.response_key
            session.last_response_payload = result.response_payload
            session.turn_count += 1

            if self.nlu_logger.enabled:
                self._log_nlu_and_turn(
                    session=session,
                    state_before=state_before,
                    context_snapshot=context_snapshot,
                    nlu=nlu,
                    result=result,
                    response_key=result.response_key,
                )

            self._trace_set_attr(trace, "response_key", result.response_key)

            return TurnOutput(
                response_key=result.response_key,
                response_payload=result.response_payload,
            )

        resume_key, resume_payload = resume
        combined_payload = {
            "interrupt_response_key": result.response_key,
            "interrupt_response_payload": result.response_payload,
            "resume_response_key": resume_key,
            "resume_response_payload": resume_payload,
        }

        session.last_response_key = "readonly_interrupt_with_resume"
        session.last_response_payload = combined_payload
        session.turn_count += 1

        if self.nlu_logger.enabled:
            self._log_nlu_and_turn(
                session=session,
                state_before=state_before,
                context_snapshot=context_snapshot,
                nlu=nlu,
                result=result,
                response_key="readonly_interrupt_with_resume",
            )

        self._trace_set_attr(trace, "response_key", "readonly_interrupt_with_resume")

        return TurnOutput(
            response_key="readonly_interrupt_with_resume",
            response_payload=combined_payload,
        )

    def _readonly_interrupt_handler_name(self, intent: Intent) -> str | None:
        if intent == Intent.ASK_PRICE:
            return "ask_price_handler"

        if intent in {
            Intent.ASK_ITEM_INFO,
            Intent.ASK_MENU_INFO,
            Intent.ASK_OPTIONS,
            Intent.AVAILABILITY_QUERY,
            Intent.BROWSE_MENU,
            Intent.BROWSE_CATEGORY,
            Intent.RECOMMENDATION_QUERY,
            Intent.SHOW_MENU,
        }:
            return "ask_menu_info_handler"

        if intent in {Intent.SHOW_CART, Intent.SHOW_TOTAL}:
            return "cart_handler"

        return None

    def _apply_command(self, session: Session, command: dict[str, Any]) -> dict[str, Any]:
        cmd_type = command.get("type")
        payload = command.get("payload") or {}

        if cmd_type == "ADD_ITEM_TO_CART":
            from app.cart.cart_item import CartItem

            cart_item = CartItem.create(
                item_id=payload["item_id"],
                quantity=payload["quantity"],
                variant_id=payload.get("variant_id"),
                sides=payload.get("sides", {}),
                side_variants=payload.get("side_variants", {}),
                modifiers=payload.get("modifiers", {}),
            )
            session.cart.add_item(cart_item)
            return {"ok": True}

        if cmd_type == "CLEAR_CART":
            session.cart.clear()
            return {"ok": True}

        if cmd_type == "REMOVE_ITEM_FROM_CART":
            session.cart.remove_item(payload["cart_item_id"])
            return {"ok": True}

        if cmd_type == "SEND_SMS":
            template = payload["template"]

            sms_request = SmsSendRequest(
                template=template,
                phone_number=payload["phone_number"],
                order_number=payload.get("order_number", ""),
                link=payload.get("link", ""),
                area=payload.get("area", ""),
            )

            sms_result: SmsSendResult | None = None
            attempts_made = 0

            for _ in range(2):
                attempts_made += 1
                sms_result = self.sms_service.send(sms_request)
                if sms_result.ok:
                    break

            return {
                "ok": bool(sms_result and sms_result.ok),
                "sid": sms_result.sid if sms_result else None,
                "error_code": sms_result.error_code if sms_result else "sms_send_failed",
                "error_message": sms_result.error_message if sms_result else "SMS send failed.",
                "template": template,
                "attempts_made": attempts_made,
            }

        if cmd_type == "transfer_call":
            return {
                "ok": True,
                "transport_only": True,
                "transfer_number": command.get("transfer_number"),
            }

        raise ValueError(f"Unknown command type: {cmd_type}")

    def _safe_session_id(self, session: Session) -> str:
        value = getattr(session, "session_id", None)
        if value is not None:
            return str(value)

        value = getattr(session, "id", None)
        if value is not None:
            return str(value)

        return "unknown_session"

    def _safe_pending_action_from_context(self, context: Any) -> str:
        action = getattr(context, "pending_action", None)
        return action.value if action is not None else ""

    def _safe_current_prompt_field_from_context(self, context: Any) -> str:
        return getattr(context, "current_prompt_field", None) or ""

    def _safe_current_item_id_from_context(self, context: Any) -> str:
        return getattr(context, "current_item_id", None) or ""

    def _safe_current_item_name_from_context(self, context: Any) -> str:
        return getattr(context, "current_item_name", None) or ""

    def _snapshot_context_for_logging(self, session: Session) -> dict[str, str]:
        ctx = session.conversation_context
        return {
            "pending_action": self._safe_pending_action_from_context(ctx),
            "current_prompt_field": self._safe_current_prompt_field_from_context(ctx),
            "current_item_id": self._safe_current_item_id_from_context(ctx),
            "current_item_name": self._safe_current_item_name_from_context(ctx),
        }

    def _log_nlu_and_turn(
        self,
        *,
        session: Session,
        state_before: ConversationState,
        context_snapshot: dict[str, str],
        nlu: Any,
        result: HandlerResult | None,
        response_key: str,
    ) -> None:
        try:
            slot_values = getattr(nlu, "slots", ()) or ()

            self.nlu_logger.log_turn(
                session_id=self._safe_session_id(session),
                turn_index=session.turn_count,
                state_before=state_before.value,
                state_after=session.conversation_state.value,
                pending_action=context_snapshot["pending_action"],
                current_prompt_field=context_snapshot["current_prompt_field"],
                current_item_id=context_snapshot["current_item_id"],
                current_item_name=context_snapshot["current_item_name"],
                user_text=session.conversation_context.last_user_text or "",
                normalized_text=getattr(nlu, "normalized_text", "") or "",
                pred_main_intent=getattr(nlu, "model_main_intent", "") or "",
                pred_sub_intent=getattr(nlu, "model_sub_intent", "") or "",
                pred_intent=getattr(getattr(nlu, "effective_intent", None), "value", "") or "",
                pred_intent_confidence=getattr(nlu, "intent_confidence", None),
                slot_model_ran=bool(getattr(nlu, "slot_model_ran", False)),
                response_key=response_key,
                command=result.command if result else None,
                slots=slot_values,
            )
        except Exception as exc:
            print(f"[NLU_CSV_LOGGER_ERROR] {type(exc).__name__}: {exc}")

    def _apply_idle_shortcuts(
        self,
        session: Session,
        intent_result: IntentResult,
    ) -> tuple[IntentResult, TurnOutput | None]:
        if session.conversation_state != ConversationState.IDLE:
            return intent_result, None

        checkout_like_intents = {
            Intent.DENY,
            Intent.END_ADDING,
            Intent.START_ORDER,
            Intent.CHECKOUT,
            Intent.CONFIRM_ORDER,
            Intent.FINISH_ORDER,
            Intent.PAYMENT_REQUEST,
            Intent.REVIEW_ORDER,
        }

        if intent_result.intent not in checkout_like_intents:
            return intent_result, None

        if not session.cart.is_empty():
            return (
                IntentResult(
                    intent=Intent.START_ORDER,
                    raw_text=intent_result.raw_text,
                ),
                None,
            )

        return (
            intent_result,
            TurnOutput(
                response_key="idle_nothing_to_checkout",
                response_payload=None,
            ),
        )

    def _rewrite_confirming_order_to_idle_if_needed(
        self,
        *,
        session: Session,
        intent: Intent,
    ) -> ConversationState:
        if session.conversation_state != ConversationState.CONFIRMING_ORDER:
            return session.conversation_state

        if intent in CONFIRMING_ORDER_EXIT_TO_IDLE_INTENTS:
            return ConversationState.IDLE

        return session.conversation_state

    def _rewrite_idle_unknown_menu_followup(
        self,
        *,
        session: Session,
        intent_result: IntentResult,
        normalized_text: str,
    ) -> IntentResult:
        if session.conversation_state != ConversationState.IDLE:
            return intent_result

        if intent_result.intent != Intent.UNKNOWN:
            return intent_result

        if not normalized_text:
            return intent_result

        last_key = getattr(session, "last_response_key", None) or ""
        if last_key not in {
            "show_category",
            "menu_ambiguity",
            "show_menu_categories",
            "show_item_info",
            "show_item_availability",
        }:
            return intent_result

        result = self.menu_repo.resolve_menu_query_normalized(normalized_text, limit=5)
        if result.type in {
            MenuQueryType.ITEM,
            MenuQueryType.ITEM_AMBIGUOUS,
            MenuQueryType.CATEGORY,
            MenuQueryType.CATEGORY_SINGLE_ITEM,
            MenuQueryType.CATEGORY_AMBIGUOUS,
        }:
            return IntentResult(
                intent=Intent.ASK_ITEM_INFO,
                raw_text=intent_result.raw_text,
            )

        if last_key in {"show_category", "menu_ambiguity", "show_menu_categories"}:
            return IntentResult(
                intent=Intent.ASK_ITEM_INFO,
                raw_text=intent_result.raw_text,
            )

        return intent_result

    def _trace_set_initial_fields(
        self,
        *,
        trace: Any | None,
        session: Session,
        user_text: str,
        state_before: ConversationState,
        engine_start_monotonic: float,
    ) -> None:
        ctx = session.conversation_context
        self._trace_set_attr(trace, "session_id", self._safe_session_id(session))
        self._trace_set_attr(trace, "user_text", user_text)
        self._trace_set_attr(trace, "state_before", state_before.value)
        self._trace_set_attr(trace, "pending_action", self._safe_pending_action_from_context(ctx))
        self._trace_set_attr(trace, "current_prompt_field", self._safe_current_prompt_field_from_context(ctx))
        self._trace_set_attr(trace, "current_item_id", self._safe_current_item_id_from_context(ctx))
        self._trace_set_attr(trace, "current_item_name", self._safe_current_item_name_from_context(ctx))
        self._trace_set_attr(trace, "engine_start_monotonic", engine_start_monotonic)

    def _trace_set_nlu_fields(self, *, trace: Any | None, nlu: Any) -> None:
        normalized_text = getattr(nlu, "normalized_text", "") or ""
        effective_intent = getattr(getattr(nlu, "effective_intent", None), "value", "") or ""

        self._trace_set_attr(trace, "normalized_text", normalized_text)
        self._trace_set_attr(trace, "pred_main_intent", getattr(nlu, "model_main_intent", "") or "")
        self._trace_set_attr(trace, "pred_sub_intent", getattr(nlu, "model_sub_intent", "") or "")
        self._trace_set_attr(trace, "pred_intent", effective_intent)
        self._trace_set_attr(trace, "pred_intent_confidence", getattr(nlu, "intent_confidence", None))
        self._trace_set_attr(trace, "stt_final_text_chars", len(normalized_text))

        slots = getattr(nlu, "slots", ()) or ()
        slot_names: list[str] = []
        slot_values: list[str] = []

        for slot in slots:
            label = getattr(slot, "label", None) or getattr(slot, "name", None)
            value = getattr(slot, "value", None)

            if isinstance(slot, dict):
                label = label or slot.get("label") or slot.get("name")
                value = value if value is not None else slot.get("value")

            if label:
                slot_names.append(str(label))
            if value is not None:
                slot_values.append(str(value))

        self._trace_set_attr(trace, "slot_names", slot_names)
        self._trace_set_attr(trace, "slot_values", slot_values)
        self._trace_set_attr(trace, "intent_model_ms", getattr(nlu, "intent_model_ms", None))
        self._trace_set_attr(trace, "slot_model_ms", getattr(nlu, "slot_model_ms", None))

    def _trace_finalize(
        self,
        *,
        trace: Any | None,
        session: Session,
        response_key: str,
        total_start_monotonic: float,
        total_ms: float,
        preprocess_ms: float,
        nlu_ms: float,
        flow_ms: float,
        route_ms: float,
        handler_ms: float,
        command: dict[str, Any] | None = None,
    ) -> None:
        engine_end = time.perf_counter()
        ctx = session.conversation_context

        self._trace_set_attr(trace, "response_key", response_key)
        self._trace_set_attr(trace, "state_after", session.conversation_state.value)
        self._trace_set_attr(trace, "pending_action", self._safe_pending_action_from_context(ctx))
        self._trace_set_attr(trace, "current_prompt_field", self._safe_current_prompt_field_from_context(ctx))
        self._trace_set_attr(trace, "current_item_id", self._safe_current_item_id_from_context(ctx))
        self._trace_set_attr(trace, "current_item_name", self._safe_current_item_name_from_context(ctx))
        self._trace_set_attr(trace, "engine_end_monotonic", engine_end)
        self._trace_set_attr(trace, "turn_total_ms", round(total_ms, 3))
        self._trace_set_attr(trace, "preprocess_ms", round(preprocess_ms, 3))
        self._trace_set_attr(trace, "nlu_ms", round(nlu_ms, 3))
        self._trace_set_attr(trace, "flow_ms", round(flow_ms, 3))
        self._trace_set_attr(trace, "route_ms", round(route_ms, 3))
        self._trace_set_attr(trace, "handler_ms", round(handler_ms, 3))
        self._trace_set_attr(trace, "engine_total_ms", round((engine_end - total_start_monotonic) * 1000.0, 3))
        if command is not None:
            self._trace_set_attr(trace, "command", command)

    def _trace_set_attr(self, trace: Any | None, attr_name: str, value: Any) -> None:
        if trace is None:
            return
        try:
            setattr(trace, attr_name, value)
        except Exception:
            return

    def _handle_auto_payment_check(self, session: Session) -> TurnOutput:
        """Verify payment status without going through the NLU pipeline.

        Called when the transport layer fires the ``__auto_payment_check__``
        sentinel.  Only meaningful in WAITING_FOR_PAYMENT and
        WAITING_FOR_CHECKOUT_COMPLETION states; all other states return a
        silent no-op so the call flow is never disrupted.
        """
        ctx = session.conversation_context
        state = session.conversation_state

        if state == ConversationState.WAITING_FOR_PAYMENT:
            pending_state = ConversationState.WAITING_FOR_PAYMENT
            pending_key = "waiting_for_payment"
        elif state == ConversationState.WAITING_FOR_CHECKOUT_COMPLETION:
            pending_state = ConversationState.WAITING_FOR_CHECKOUT_COMPLETION
            pending_key = "waiting_for_checkout_completion"
        else:
            # State changed between scheduling and firing — do nothing.
            return self._hydrate_output(
                session=session,
                output=TurnOutput(response_key=session.last_response_key or "waiting_for_payment"),
            )

        delivery = ctx.delivery_address
        order_number = getattr(delivery, "order_number", None)

        result = verify_payment_for_order(
            checkout_service=self.checkout_service,
            order_number=order_number,
            pending_state=pending_state,
            pending_response_key=pending_key,
        )

        if result.reset_context:
            ctx.reset()
        if result.command:
            self._apply_command(session, result.command)

        session.conversation_state = result.next_state
        self._apply_session_response(
            session=session,
            intent=Intent.PAYMENT_STATUS,
            response_key=result.response_key,
            response_payload=result.response_payload,
        )

        return self._hydrate_output(
            session=session,
            output=TurnOutput(
                response_key=result.response_key,
                response_payload=result.response_payload,
                end_call_after_playback=(result.next_state == ConversationState.COMPLETED),
            ),
        )

    def _order_type_required(self, session: Session) -> bool:
        ctx = session.conversation_context
        return ctx.order_type not in {"pickup", "delivery"}

    def _normalize_order_type_gate_state(self, session: Session) -> None:
        if session.conversation_state == ConversationState.COMPLETED:
            return
        # Don't clobber the device-type gate or the human-handoff state.
        if session.conversation_state in {
            ConversationState.WAITING_FOR_CALLER_DEVICE_TYPE,
            ConversationState.WAITING_FOR_LANDLINE_PICKUP_CONFIRMATION,
            ConversationState.TRANSFERRING_TO_HUMAN_AGENT,
        }:
            return
        if self._order_type_required(session):
            session.conversation_state = ConversationState.WAITING_FOR_ORDER_TYPE

    def _combine_order_type_and_followup_response(
        self,
        *,
        order_type_key: str,
        followup_output: TurnOutput,
    ) -> TurnOutput:
        prefix = "Got it. Pickup." if order_type_key == "order_type_captured_pickup" else "Got it. Delivery."

        if followup_output.spoken_response_text:
            spoken = f"{prefix} {followup_output.spoken_response_text}"
        else:
            spoken = None

        internal = spoken

        return TurnOutput(
            response_key=followup_output.response_key,
            response_payload=followup_output.response_payload,
            internal_response_text=internal,
            spoken_response_text=spoken,
        )

    @staticmethod
    def _normalize_response_text(text: str | None) -> str:
        return " ".join((text or "").split()).strip()

    def _hydrate_output(
        self,
        *,
        session: Session,
        output: TurnOutput,
    ) -> TurnOutput:
        internal_text = self._normalize_response_text(
            output.internal_response_text
            or self.responder.build(
                response_key=output.response_key,
                context=session.conversation_context,
                payload=output.response_payload,
            )
        )

        spoken_text = self._normalize_response_text(
            output.spoken_response_text or internal_text
        )

        return TurnOutput(
            response_key=output.response_key,
            response_payload=output.response_payload,
            internal_response_text=internal_text,
            spoken_response_text=spoken_text,
            end_call_after_playback=output.end_call_after_playback,
            transfer_call_to_number=output.transfer_call_to_number,
        )
