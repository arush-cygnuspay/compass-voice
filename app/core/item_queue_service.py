"""Multi-item queue draining service.

After an item is added to the cart and the FSM returns to IDLE, this
service inspects the pending-item queue (from a multi-item utterance)
and starts the next item's add-item flow. Behavior moved verbatim from
``turn_engine.py``.
"""
from __future__ import annotations

from typing import Any

from app.core.command_executor import CommandExecutor
from app.session.session import Session
from app.state_machine.handler_result import HandlerResult
from app.state_machine.models.conversation_state import ConversationState


class ItemQueueService:
    """Public method: ``try_drain``. Owns no state of its own beyond
    the references injected at construction time."""

    def __init__(
        self,
        *,
        handlers: dict[str, Any],
        command_executor: CommandExecutor,
    ) -> None:
        self.handlers = handlers
        self.command_executor = command_executor

    def try_drain(
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
            self.command_executor.execute(session, next_result.command)

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
            deeper = self.try_drain(
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
