# app/state_machine/handlers/item/add_item/group_resolution_handler.py
"""Template-method base for group-style selection handlers.

Provides the shared ``_step_to_result`` / ``_match_debug_payload`` skeleton
used by WaitingForModifierHandler, WaitingForSideHandler, and
WaitingForSizeHandler, removing ~90 lines of byte-for-byte duplication.

Only orchestration logic that is identical across all three handlers lives
here.  Handler-specific resolution (resolver type, slot key, control-intent
responses, overflow, skip policy) stays in each subclass.
"""
from __future__ import annotations

from app.nlu.modifier_instructions import speak as _speak_modifier
from app.state_machine.handler_result import HandlerResult
from app.state_machine.handlers.base_handler import BaseHandler
from app.state_machine.handlers.item.add_item.add_item_flow import ReadyToFinalize
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState


def _spoken_modifiers_for(context: ConversationContext) -> list[str]:
    """Return modifier selections rendered as spoken phrases.

    Mirrors confirmation_decision_helper._spoken_modifiers_for so any path
    that emits item_added_successfully voices the same "no/extra/light/...
    on the side" clause.  The action and instruction live on each
    ModifierSelection — formatting is centralized in
    app.nlu.modifier_instructions.speak.
    """
    pending = context.pending_add_item
    if pending is None:
        return []
    spoken: list[str] = []
    for group in pending.modifier_groups:
        for selection in context.selected_modifier_groups.get(group.group_id, []):
            phrase = _speak_modifier(
                selection.name,
                action=selection.action,
                instruction=selection.instruction,
            )
            if phrase:
                spoken.append(phrase)
    return spoken


class GroupResolutionHandler(BaseHandler):
    """Base for handlers that resolve user input against a group of options.

    Subclasses must implement ``handle()`` (the ``BaseHandler`` contract) and
    may override any helper if their field has a genuinely different shape.

    The helpers here are intentionally narrow: they only extract the patterns
    that were literally identical across all three concrete handlers.
    """

    # ------------------------------------------------------------------
    # Shared step → HandlerResult conversion
    # ------------------------------------------------------------------

    def _step_to_result(
        self,
        context: ConversationContext,
        step,
        *,
        matched_names: list[str] | None = None,
        unmatched_names: list[str] | None = None,
        match_debug: dict[str, object] | None = None,
    ) -> HandlerResult:
        """Convert an add-item step into a HandlerResult.

        When *step* is ``ReadyToFinalize``, emit ``item_added_successfully``
        with the standard payload.  Otherwise delegate to the step's own
        ``next_state`` / ``response_key`` / ``response_payload``.

        The optional *matched_names*, *unmatched_names*, and *match_debug*
        kwargs are merged into the payload only when provided, preserving
        ``None`` on the base step payload for callers that do not pass them
        (e.g. ``WaitingForSizeHandler``).
        """
        pending = context.pending_add_item
        if pending is None:
            return HandlerResult(
                next_state=ConversationState.ERROR_RECOVERY,
                response_key="item_context_missing",
            )

        if isinstance(step, ReadyToFinalize):
            payload: dict[str, object] = {
                "item_name": pending.item_name,
                "quantity": context.quantity or 1,
                # Voice the customer's modifier choices back in the success
                # line ("...with no onions and extra cheese added").  Built
                # from the cart selections via the canonical speech helper.
                "spoken_modifiers": _spoken_modifiers_for(context),
            }
            if matched_names:
                payload["matched_names"] = matched_names
            if unmatched_names:
                payload["unmatched_names"] = unmatched_names
            payload.update(self._match_debug_payload(match_debug))
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="item_added_successfully",
                response_payload=payload,
                command=step.command.to_dict(),
                reset_context=True,
            )

        # Only copy-and-mutate the step payload when there is something to
        # inject, so callers that pass nothing still get the original payload
        # reference (which may legitimately be None).
        if matched_names or unmatched_names or match_debug:
            payload = dict(step.response_payload or {})
            if matched_names:
                payload["matched_names"] = matched_names
            if unmatched_names:
                payload["unmatched_names"] = unmatched_names
            payload.update(self._match_debug_payload(match_debug))
            return HandlerResult(
                next_state=step.next_state,
                response_key=step.response_key,
                response_payload=payload,
            )

        return HandlerResult(
            next_state=step.next_state,
            response_key=step.response_key,
            response_payload=step.response_payload,
        )

    # ------------------------------------------------------------------
    # Shared debug-payload helper
    # ------------------------------------------------------------------

    @staticmethod
    def _match_debug_payload(
        match_debug: dict[str, object] | None,
    ) -> dict[str, object]:
        """Return a shallow copy of *match_debug*, or ``{}`` when absent."""
        return dict(match_debug or {})
