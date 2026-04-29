"""Flow-policy / shortcut / rewrite helpers extracted from TurnEngine.

Owns the "before-handler" decisions that mutate session state based on
context. Behavior moved verbatim from ``turn_engine.py``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.core.payment_flow_orchestrator import PaymentFlowOrchestrator
from app.core.session_response_writer import SessionResponseWriter
from app.core.turn_diagnostics import TurnDiagnostics
from app.menu.query_result import MenuQueryType
from app.menu.repository import MenuRepository
from app.nlu.control_decision_service import DEFAULT_SERVICE as _control_service
from app.nlu.intent_resolution.intent import Intent
from app.nlu.intent_resolution.intent_result import IntentResult
from app.nlu.nlu_result import NLUResult
from app.session.session import Session
from app.state_machine.handler_result import HandlerResult
from app.state_machine.handlers.order.prepayment_correction_support import (
    clone_cart_item_with_quantity,
    extract_requested_quantity,
    resolve_cart_item_for_quantity_change,
)
from app.state_machine.handlers.system.waiting_for_caller_device_type_handler import (
    HUMAN_AGENT_TRANSFER_NUMBER,
)
from app.state_machine.models.conversation_state import ConversationState
from app.state_machine.phase3_controls import (
    is_repeat_order_summary_request,
    is_restart_item_request,
    is_restart_order_request,
)
from app.state_machine.resume_prompt_builder import ResumePromptBuilder
from app.state_machine.semantic_signals import (
    is_confirmation_reject_response,
    is_done_like_response,
)

if TYPE_CHECKING:
    from app.core.turn_engine import TurnOutput


CONFIRMING_ORDER_EXIT_TO_IDLE_INTENTS: set[Intent] = {
    Intent.ASK_ITEM_INFO,
    Intent.ASK_MENU_INFO,
    Intent.ASK_OPTIONS,
    Intent.AVAILABILITY_QUERY,
    Intent.BROWSE_MENU,
    Intent.BROWSE_CATEGORY,
    Intent.RECOMMENDATION_QUERY,
    Intent.SHOW_MENU,
}


class FlowGate:
    """Composes the flow-policy / shortcut / rewrite decisions previously
    inlined on TurnEngine."""

    def __init__(
        self,
        *,
        handlers: dict[str, Any],
        menu_repo: MenuRepository,
        cart_summary_builder: Any,
        response_writer: SessionResponseWriter,
        diagnostics: TurnDiagnostics,
        payment_flow: PaymentFlowOrchestrator,
        resume_prompt_builder: ResumePromptBuilder,
    ) -> None:
        self.handlers = handlers
        self.menu_repo = menu_repo
        self.cart_summary_builder = cart_summary_builder
        self.response_writer = response_writer
        self.diagnostics = diagnostics
        self.payment_flow = payment_flow
        self.resume_prompt_builder = resume_prompt_builder

    def _handle_readonly_interrupt(
        self,
        *,
        session: Session,
        state_before: ConversationState,
        intent_result: IntentResult,
        nlu: Any,
        trace: Any | None = None,
    ) -> "TurnOutput | None":
        from app.core.turn_engine import TurnOutput

        handler_name = self._readonly_interrupt_handler_name(intent_result.intent)
        if handler_name is None:
            return None

        handler = self.handlers.get(handler_name)
        if handler is None:
            raise KeyError(f"Handler not registered: {handler_name}")

        preserved_state = session.conversation_state

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

            self.diagnostics._trace_set_attr(trace, "response_key", result.response_key)

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

        self.diagnostics._trace_set_attr(trace, "response_key", "readonly_interrupt_with_resume")

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

    def _handle_phase3_control_shortcuts(
        self,
        *,
        session: Session,
        state_before: ConversationState,
        intent_result: IntentResult,
        nlu: Any,
    ) -> "TurnOutput | None":
        from app.core.turn_engine import TurnOutput

        text = getattr(nlu, "normalized_text", "") or ""
        ctx = session.conversation_context

        nlu_result: NLUResult | None = nlu if isinstance(nlu, NLUResult) else None
        if _control_service.resolve_agent_request(text, nlu_result).intent == Intent.REQUEST_AGENT:
            self.payment_flow._log_payment_event(
                session=session,
                state=state_before,
                event_name="user_requested_agent",
                metadata={"reason": "direct_request"},
            )
            return TurnOutput(
                response_key="transferring_to_human_agent",
                response_payload={
                    "transfer_number": HUMAN_AGENT_TRANSFER_NUMBER,
                    "_payment_events": [
                        {
                            "event_name": "user_requested_agent",
                            "metadata": {"reason": "direct_request"},
                        }
                    ],
                },
                end_call_after_playback=True,
                transfer_call_to_number=HUMAN_AGENT_TRANSFER_NUMBER,
            )

        if state_before in {
            ConversationState.CONFIRMING_ITEM,
            ConversationState.WAITING_FOR_SIDE,
            ConversationState.WAITING_FOR_SIDE_SIZE,
            ConversationState.WAITING_FOR_MODIFIER,
            ConversationState.WAITING_FOR_SIZE,
            ConversationState.WAITING_FOR_QUANTITY,
        } and is_restart_item_request(text):
            ctx.reset_task()
            session.conversation_state = ConversationState.IDLE
            return TurnOutput(response_key="current_item_restarted")

        if state_before in {
            ConversationState.CONFIRMING_ORDER,
            ConversationState.WAITING_FOR_PAYMENT,
            ConversationState.WAITING_FOR_CHECKOUT_COMPLETION,
        }:
            if is_restart_order_request(text):
                ctx.reset_task()
                ctx.resume_order_confirmation_after_edit = False
                session.cart.clear()
                self.payment_flow._reset_payment_wait_tracking(session)
                session.conversation_state = ConversationState.IDLE
                return TurnOutput(response_key="order_restart_ready")

            if is_repeat_order_summary_request(text):
                session.conversation_state = ConversationState.CONFIRMING_ORDER
                return TurnOutput(
                    response_key="confirm_order_summary",
                    response_payload=self.cart_summary_builder.build(session.cart),
                )

            correction_output = self._handle_prepayment_quantity_correction(
                session=session,
                state_before=state_before,
                intent_result=intent_result,
                normalized_text=text,
                nlu_result=nlu_result,
            )
            if correction_output is not None:
                return correction_output

            if intent_result.intent in {
                Intent.ADD_ITEM,
                Intent.REMOVE_ITEM,
                Intent.REPLACE_ITEM,
                Intent.MODIFY_ITEM,
                Intent.UNDO_LAST,
                Intent.CHANGE_QUANTITY,
            }:
                ctx.resume_order_confirmation_after_edit = True
                self.payment_flow._reset_payment_wait_tracking(session)
                self.payment_flow._log_payment_event(
                    session=session,
                    state=state_before,
                    event_name="order_corrected_before_payment",
                    metadata={"intent": intent_result.intent.value},
                )
                session.conversation_state = ConversationState.IDLE

        return None

    def _handle_prepayment_quantity_correction(
        self,
        *,
        session: Session,
        state_before: ConversationState,
        intent_result: IntentResult,
        normalized_text: str,
        nlu_result: NLUResult | None = None,
    ) -> "TurnOutput | None":
        from app.core.turn_engine import TurnOutput

        if (
            _control_service.resolve_quantity_correction(normalized_text, nlu_result).intent
            != Intent.CHANGE_QUANTITY
        ):
            return None

        quantity = extract_requested_quantity(normalized_text)
        if quantity is None or quantity <= 0:
            return None

        cart_item = resolve_cart_item_for_quantity_change(
            menu_repo=self.menu_repo,
            session=session,
            context=session.conversation_context,
            user_text=normalized_text,
        )
        if cart_item is None:
            return None

        replacement = clone_cart_item_with_quantity(cart_item, quantity)
        if not session.cart.replace_item(cart_item.cart_item_id, replacement):
            return None

        self.payment_flow._reset_payment_wait_tracking(session)
        self.payment_flow._log_payment_event(
            session=session,
            state=state_before,
            event_name="order_corrected_before_payment",
            metadata={
                "intent": "change_quantity",
                "item_id": cart_item.item_id,
                "quantity": quantity,
            },
        )
        session.conversation_state = ConversationState.CONFIRMING_ORDER
        return TurnOutput(
            response_key="confirm_order_summary",
            response_payload={
                **self.cart_summary_builder.build(session.cart),
                "updated_order": True,
            },
        )

    def _apply_idle_shortcuts(
        self,
        session: Session,
        intent_result: IntentResult,
    ) -> "tuple[IntentResult, TurnOutput | None]":
        from app.core.turn_engine import TurnOutput

        if session.conversation_state != ConversationState.IDLE:
            return intent_result, None

        last_key = getattr(session, "last_response_key", None) or ""
        if (
            last_key in {"item_added_successfully", "item_removed_successfully"}
            and intent_result.intent == Intent.UNKNOWN
            and (
                is_done_like_response(Intent.UNKNOWN, intent_result.raw_text)
                or is_confirmation_reject_response(Intent.UNKNOWN, intent_result.raw_text)
            )
        ):
            return (
                IntentResult(
                    intent=Intent.START_ORDER,
                    raw_text=intent_result.raw_text,
                ),
                None,
            )

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
        followup_output: "TurnOutput",
    ) -> "TurnOutput":
        from app.core.turn_engine import TurnOutput

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
