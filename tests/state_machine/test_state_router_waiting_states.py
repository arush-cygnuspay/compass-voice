from app.nlu.intent_resolution.intent import Intent
from app.nlu.intent_resolution.intent_result import IntentResult
from app.state_machine.conversation_state import ConversationState
from app.state_machine.state_router import StateRouter


def test_waiting_for_size_routes_to_waiting_size_handler_even_for_other_intent():
    router = StateRouter()

    result = router.route(
        ConversationState.WAITING_FOR_SIZE,
        IntentResult(intent=Intent.ASK_PRICE, raw_text="how much"),
    )

    assert result.allowed is True
    assert result.handler_name == "waiting_for_size_handler"