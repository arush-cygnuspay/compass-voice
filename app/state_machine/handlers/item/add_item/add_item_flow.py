# app/state_machine/handlers/item/add_item/add_item_flow.py
from app.menu.models import MenuItem
from app.state_machine.context import ConversationContext
from app.state_machine.conversation_state import ConversationState


def determine_next_add_item_state(
    item: MenuItem,
    context: ConversationContext,
) -> ConversationState:
    """
    Decides the next step in add-item flow based on menu + context.
    """

    # 1️⃣ Size / variant
    if item.pricing.mode == "variant" and not context.selected_variant_id:
        return ConversationState.WAITING_FOR_SIZE

    # 2️⃣ Required side groups
    for group in item.side_groups:
        if group.is_required and group.group_id not in context.selected_side_groups:
            return ConversationState.WAITING_FOR_SIDE

    # 3️⃣ Modifiers (optional but still a step)
    if item.modifier_groups and not context.selected_modifier_groups:
        return ConversationState.WAITING_FOR_MODIFIER

    # 4️⃣ Quantity
    if context.quantity is None:
        return ConversationState.WAITING_FOR_QUANTITY

    # 5️⃣ Ready to finalize
    return ConversationState.ADDING_ITEM

def _response_key_for_state(state: ConversationState) -> str:
    return {
        ConversationState.WAITING_FOR_SIDE: "ask_for_side",
        ConversationState.WAITING_FOR_SIZE: "ask_for_size",
        ConversationState.WAITING_FOR_MODIFIER: "ask_for_modifier",
        ConversationState.WAITING_FOR_QUANTITY: "ask_for_quantity",
        ConversationState.ADDING_ITEM: "ready_to_add_item",
    }.get(state, "unhandled_state")

def determine_next_add_item_state(item: MenuItem, context: ConversationContext) -> ConversationState:
    # 1️⃣ Item size FIRST
    if item.pricing.mode == "variant":
        context.size_target = {"type": "item"}
        return ConversationState.WAITING_FOR_SIZE

    # 2️⃣ Then sides
    if item.side_groups:
        return ConversationState.WAITING_FOR_SIDE

    # 3️⃣ Then modifiers
    if item.modifier_groups:
        return ConversationState.WAITING_FOR_MODIFIER

    # 4️⃣ Finally quantity
    return ConversationState.WAITING_FOR_QUANTITY

