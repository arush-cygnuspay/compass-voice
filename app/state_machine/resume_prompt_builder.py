# app/state_machine/resume_prompt_builder.py
from __future__ import annotations

from dataclasses import dataclass, field as dc_field

from app.session.session import Session
from app.state_machine.models.conversation_state import ConversationState


@dataclass
class ResumePromptPayload:
    """Snapshot of the context needed to re-render a pending prompt.

    Populated by ResumePromptBuilder.build() at interrupt time so the
    stored last_response_payload can render the resume question without
    re-deriving from context.  This matters when context is partially
    stale after a reconnect.
    """

    current_item_name: str | None = None
    field: str | None = None
    group: str | None = None
    top_choices: list[str] = dc_field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "current_item_name": self.current_item_name,
            "field": self.field,
            "group": self.group,
            "top_choices": list(self.top_choices),
        }


def _top_choices_from_context(context, k: int = 4) -> list[str]:
    """Return up to k choice names already stored on context."""
    values = getattr(context, "available_choices_values", None) or ()
    return [str(v) for v in values[:k] if v]


class ResumePromptBuilder:
    """Reconstruct the pending question for the active flow
    without mutating session state."""

    def build(self, session: Session) -> tuple[str, dict] | None:
        state = session.conversation_state
        context = session.conversation_context

        if state == ConversationState.WAITING_FOR_SIZE:
            payload = ResumePromptPayload(
                current_item_name=context.current_item_name or "this item",
                field="size",
                group=None,
                top_choices=_top_choices_from_context(context),
            )
            return "ask_for_size", payload.to_dict()

        if state == ConversationState.WAITING_FOR_SIDE:
            payload = ResumePromptPayload(
                current_item_name=context.current_item_name or "this item",
                field="side",
                group=None,
                top_choices=_top_choices_from_context(context),
            )
            return "ask_for_side", payload.to_dict()

        if state == ConversationState.WAITING_FOR_SIDE_SIZE:
            side_item_name = (
                getattr(context, "pending_side_item_name", None)
                or context.current_item_name
                or "this side"
            )
            choices = _top_choices_from_context(context)
            payload = ResumePromptPayload(
                current_item_name=side_item_name,
                field="side_size",
                group=None,
                top_choices=choices,
            )
            d = payload.to_dict()
            # Include the keys that _ask_for_side_size expects
            d["side_item_name"] = side_item_name
            d["available_sizes"] = list(
                getattr(context, "available_choices_values", None) or ()
            )
            return "ask_for_side_size", d

        if state == ConversationState.WAITING_FOR_MODIFIER:
            payload = ResumePromptPayload(
                current_item_name=context.current_item_name or "this item",
                field="modifier",
                group=None,
                top_choices=_top_choices_from_context(context),
            )
            return "ask_for_modifier", payload.to_dict()

        if state == ConversationState.WAITING_FOR_QUANTITY:
            item_name = context.current_item_name or "this item"
            payload = ResumePromptPayload(
                current_item_name=item_name,
                field="quantity",
                group=None,
                top_choices=[],
            )
            d = payload.to_dict()
            d["item_name"] = item_name  # key expected by ask_item_quantity
            return "ask_for_quantity", d

        return None
