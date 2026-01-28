# app/state_machine/flow_sets.py

from app.state_machine.conversation_state import ConversationState

ADD_ITEM_FLOW_STATES = {
    ConversationState.CONFIRMING_ITEM,
    ConversationState.WAITING_FOR_SIDE,
    ConversationState.WAITING_FOR_MODIFIER,
    ConversationState.WAITING_FOR_SIZE,
    ConversationState.WAITING_FOR_QUANTITY,
}

ORDER_FLOW_STATES = {
    ConversationState.CONFIRMING_ORDER,
    ConversationState.WAITING_FOR_PAYMENT,
}
