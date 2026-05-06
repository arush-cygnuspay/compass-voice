from __future__ import annotations

from app.nlu.intent_resolution.intent import Intent
from app.session.session import Session
from app.state_machine.control_intent_resolver import (
    ControlIntentKind,
    log_control_intent_event,
    resolve_control_intent,
)
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.pending_item_models import InterruptProposal
from app.state_machine.models.conversation_state import ConversationState
from app.state_machine.handler_result import HandlerResult
from app.state_machine.handlers.base_handler import BaseHandler
from app.state_machine.handlers.item.add_item.add_item_flow import (
    ReadyToFinalize,
    determine_next_add_item_step,
)
from app.utils.quantity_detection import detect_quantity, normalize_quantity


from app.state_machine.flow_sets import SOFT_SWITCH_INTENTS


class WaitingForQuantityHandler(BaseHandler):
    """
    Resolve quantity while the add-item flow is waiting for it.

    Rules:
    - prefer the already-resolved QUANTITY slot when available
    - fallback to text-based quantity detection
    - accept short direct answers like "2" or "three"
    - if quantity is resolved, continue canonical add-item flow immediately
    - if quantity is not resolved and the user tries a different flow-level action,
      route through cancellation confirmation
    - do not aggressively scold with invalid quantity for simple item restatements;
      softly re-ask first
    """

    def handle(
        self,
        intent: Intent,
        context: ConversationContext,
        user_text: str,
        session: Session | None = None,
    ) -> HandlerResult:
        pending = context.pending_add_item

        if pending is None or not context.current_item_id:
            return HandlerResult(
                next_state=ConversationState.ERROR_RECOVERY,
                response_key="item_context_missing",
            )

        normalized_user_text = (user_text or "").strip()

        # --- Step 1: CANCEL must always win, regardless of slots. ---
        # Resolve only enough to detect an explicit cancel before we do
        # anything else; the remaining control intents are checked below,
        # after we've had a chance to consume a QUANTITY slot.
        control_intent = resolve_control_intent(
            normalized_user_text,
            intent,
            getattr(context.last_nlu, "model_sub_intent", None),
            ConversationState.WAITING_FOR_QUANTITY,
            context,
            nlu_result=context.last_nlu,
            intent_confidence=context.last_intent_confidence,
        )

        if control_intent is not None and control_intent.kind == ControlIntentKind.CANCEL:
            log_control_intent_event(
                "control_intent_action",
                state=ConversationState.WAITING_FOR_QUANTITY.value,
                action="cancel_pending_item",
                kind=control_intent.kind.value,
            )
            context.reset_item_scope()
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="item_cancelled_successfully",
            )

        # --- Step 2: QUANTITY slot / text takes priority over intent labels. ---
        # NLU intent classification is unreliable for short quantity utterances
        # ("Two.", "I said two") — these are frequently mis-labelled as
        # AFFIRM or DENY. A resolved QUANTITY slot is authoritative and must
        # win before any intent-based re-ask logic fires.
        extracted_quantity = self._extract_quantity_from_context_or_text(
            context=context,
            user_text=normalized_user_text,
        )

        if extracted_quantity is not None:
            if extracted_quantity <= 0:
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_QUANTITY,
                    response_key="invalid_quantity_option",
                    response_payload={"item_name": pending.item_name},
                )

            context.quantity = extracted_quantity

            if context.current_prompt_field == "quantity":
                context.current_prompt_field = None

            if context.available_choices_kind == "quantity":
                context.available_choices_kind = None
                context.available_choices_values = ()

            step = determine_next_add_item_step(context)
            return self._step_to_result(context, step)

        # --- Step 3: No valid quantity found — apply remaining intent routing. ---
        if control_intent is not None:
            if control_intent.kind == ControlIntentKind.OPTIONS_REQUEST:
                log_control_intent_event(
                    "options_requested",
                    state=ConversationState.WAITING_FOR_QUANTITY.value,
                    field_name="quantity",
                )
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_QUANTITY,
                    response_key="ask_for_quantity",
                    response_payload={"item_name": pending.item_name},
                )

            if control_intent.kind == ControlIntentKind.META_CLARIFY:
                log_control_intent_event(
                    "meta_clarify_repeated",
                    state=ConversationState.WAITING_FOR_QUANTITY.value,
                    field_name="quantity",
                )
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_QUANTITY,
                    response_key="ask_for_quantity",
                    response_payload={"item_name": pending.item_name},
                )

            if control_intent.kind in {
                ControlIntentKind.AFFIRM,
                ControlIntentKind.DENY,
                ControlIntentKind.DONE,
            }:
                log_control_intent_event(
                    "control_intent_action",
                    state=ConversationState.WAITING_FOR_QUANTITY.value,
                    action="quantity_still_required",
                    kind=control_intent.kind.value,
                )
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_QUANTITY,
                    response_key="ask_for_quantity",
                    response_payload={"item_name": pending.item_name},
                )

        if intent in SOFT_SWITCH_INTENTS:
            context.return_state = ConversationState.WAITING_FOR_QUANTITY
            return HandlerResult(
                next_state=ConversationState.CANCELLATION_CONFIRMATION,
                response_key="confirm_cancel_current_item_for_new_request",
                response_payload={"item_name": pending.item_name},
                awaiting_flow_confirmation=True,
                interrupt_proposal=InterruptProposal(
                    text=normalized_user_text,
                    predicted_main_intent=None,
                    predicted_sub_intent=intent.value,
                ),
            )

        if self._looks_like_item_restatement(
            user_text=normalized_user_text,
            item_name=pending.item_name,
        ):
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_QUANTITY,
                response_key="ask_for_quantity",
                response_payload={"item_name": pending.item_name},
            )

        return HandlerResult(
            next_state=ConversationState.WAITING_FOR_QUANTITY,
            response_key="ask_for_quantity",
            response_payload={"item_name": pending.item_name},
        )

    def _extract_quantity_from_context_or_text(
        self,
        *,
        context: ConversationContext,
        user_text: str,
    ) -> int | None:
        slots = getattr(context, "last_slots", ()) or ()
        for slot in slots:
            slot_name = str(getattr(slot, "name", "")).upper()
            if slot_name != "QUANTITY":
                continue

            value = getattr(slot, "value", None)

            if isinstance(value, int):
                return value

            if isinstance(value, str):
                stripped = value.strip()
                if stripped.isdigit():
                    return int(stripped)

        normalized_quantity = normalize_quantity(user_text)
        if isinstance(normalized_quantity, int):
            return normalized_quantity

        quantity_info = detect_quantity(user_text)
        if not quantity_info:
            return None

        if quantity_info.get("type") == "vague":
            return None

        value = quantity_info.get("value")
        if isinstance(value, int):
            return value

        return None

    def _looks_like_item_restatement(
        self,
        *,
        user_text: str,
        item_name: str,
    ) -> bool:
        text = (user_text or "").strip().lower()
        item = (item_name or "").strip().lower()

        if not text or not item:
            return False

        filler_prefixes = (
            "i want ",
            "i want a ",
            "i want an ",
            "i would like ",
            "i would like a ",
            "i would like an ",
            "i will take ",
            "ill take ",
            "add ",
            "give me ",
            "get me ",
            "make it ",
            "and ",
        )

        normalized = text
        changed = True
        while changed and normalized:
            changed = False
            for prefix in filler_prefixes:
                if normalized.startswith(prefix):
                    normalized = normalized[len(prefix):].strip()
                    changed = True
                    break

        if not normalized:
            return False

        if normalized == item:
            return True

        return normalized in item or item in normalized

    def _step_to_result(self, context: ConversationContext, step) -> HandlerResult:
        pending = context.pending_add_item
        if pending is None:
            return HandlerResult(
                next_state=ConversationState.ERROR_RECOVERY,
                response_key="item_context_missing",
            )

        if isinstance(step, ReadyToFinalize):
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="item_added_successfully",
                response_payload={
                    "item_name": pending.item_name,
                    "quantity": context.quantity or 1,
                },
                command=step.command.to_dict(),
                reset_context=True,
            )

        return HandlerResult(
            next_state=step.next_state,
            response_key=step.response_key,
            response_payload=step.response_payload,
        )
