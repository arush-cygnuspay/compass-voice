"""Multi-item queue draining service.

After an item is added to the cart and the FSM returns to IDLE, this
service inspects the pending-item queue (from a multi-item utterance)
and starts the next item's add-item flow. Behavior moved verbatim from
``turn_engine.py``.
"""
from __future__ import annotations

import os
from typing import Any

from app.core.command_executor import CommandExecutor
from app.session.session import Session
from app.state_machine.handler_result import HandlerResult
from app.state_machine.models.conversation_state import ConversationState

# Maximum number of queued items processed in one try_drain call.
# Items beyond this limit are rejected and logged to prevent unbounded
# iterative loops on pathological multi-item utterances.
MAX_QUEUE_DEPTH: int = int(os.getenv("COMPASS_MAX_ITEM_QUEUE_DEPTH", "20"))


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
        from a multi-item utterance. If so, dequeue and process queued
        items iteratively until one requires user input, the queue is
        empty, or MAX_QUEUE_DEPTH items have been processed.

        Returns a new HandlerResult if a queued item was started,
        or None if no queue drain is needed.

        The returned HandlerResult carries the same shape as a direct
        handler result, extended with ``queue_transition=True`` and
        optional ``chain_prev_items`` in the response_payload.
        """
        ctx = session.conversation_context

        # Only drain when item was just successfully added and we'd go to IDLE
        if current_result.response_key != "item_added_successfully":
            return None
        if current_result.next_state != ConversationState.IDLE:
            return None
        if not ctx.pending_item_queue:
            return None

        add_handler: Any = self.handlers.get("add_item_handler")
        if add_handler is None:
            return None

        # ── Overflow guard ──────────────────────────────────────────────────
        # Reject items beyond MAX_QUEUE_DEPTH before entering the drain loop
        # so the loop is bounded and recursion is never needed.
        if len(ctx.pending_item_queue) > MAX_QUEUE_DEPTH:
            overflow_count = len(ctx.pending_item_queue) - MAX_QUEUE_DEPTH
            rejected_texts = []
            while len(ctx.pending_item_queue) > MAX_QUEUE_DEPTH:
                overflow_item = ctx.pending_item_queue.pop()  # remove from tail
                rejected_texts.append(getattr(overflow_item, "raw_text", "?")[:40])
            print(
                "[ITEM_QUEUE_OVERFLOW]",
                {
                    "session_id": session.session_id,
                    "restaurant_id": session.restaurant_id,
                    "overflow_count": overflow_count,
                    "limit": MAX_QUEUE_DEPTH,
                    "rejected_items": rejected_texts,
                },
            )

        # ── Iterative drain loop ────────────────────────────────────────────
        # Processes at most MAX_QUEUE_DEPTH items.  Mirrors the behaviour of
        # the former recursive implementation exactly:
        #
        # • chain_prev_items — accumulated info for items that were
        #   instantly added during this drain pass.  The returned payload
        #   sets chain_prev_items to [chain_prev_items[0]], preserving the
        #   same "outermost-only" semantics the recursive unwinding produced.
        # • active_result    — the HandlerResult of the most recently
        #   processed item (used to derive prev_item_name / prev_quantity
        #   for the next iteration).

        active_result: HandlerResult = current_result
        chain_prev_items: list[dict[str, Any]] = []
        last_next_result: HandlerResult | None = None
        last_next_payload: dict[str, Any] = {}

        while ctx.pending_item_queue:
            next_item = ctx.pending_item_queue.popleft()
            remaining_count = len(ctx.pending_item_queue)

            prev_payload = active_result.response_payload or {}
            prev_item_name: str = prev_payload.get("item_name", "item")
            prev_quantity: Any = prev_payload.get("quantity", 1)

            # Set up context for the new item
            ctx.reset_item_scope()
            ctx.pending_action = None

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
                ctx.reset_item_scope()

            # Do NOT set session.conversation_state here — TurnEngine applies
            # it after try_drain returns via HandlerResult.next_state.

            next_payload: dict[str, Any] = dict(next_result.response_payload or {})
            next_payload["queue_transition"] = True
            next_payload["prev_item_name"] = prev_item_name
            next_payload["prev_quantity"] = prev_quantity
            next_payload["next_item_name"] = next_item.item_slot_value or next_item.raw_text
            next_payload["remaining_queue_count"] = remaining_count

            last_next_result = next_result
            last_next_payload = next_payload

            # If the item was instantly added and more remain, continue the
            # loop (replaces the former recursive call).
            if next_result.response_key == "item_added_successfully" and ctx.pending_item_queue:
                chain_prev_items.append({"name": prev_item_name, "quantity": prev_quantity})
                active_result = next_result
                continue

            # Item needs user input OR queue is now empty — stop draining.
            break

        if last_next_result is None:
            return None

        # Attach chain summary to the final payload.  The recursive version
        # always overwrote chain_prev_items with the outermost prev_item, so
        # we reproduce that by using only chain_prev_items[0] (the item that
        # was added just before the chain started).
        if chain_prev_items:
            last_next_payload["chain_prev_items"] = [chain_prev_items[0]]

        return HandlerResult(
            next_state=last_next_result.next_state,
            response_key=last_next_result.response_key,
            response_payload=last_next_payload,
            command=None,  # commands already applied inside the loop
            reset_context=False,
        )
