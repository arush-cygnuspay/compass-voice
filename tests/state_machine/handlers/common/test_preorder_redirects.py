from app.nlu.intent_resolution.intent import Intent
from app.nlu.nlu_result import SlotValue
from app.state_machine.handlers.common.preorder_redirect_utils import (
    looks_like_ordering_request,
)
from app.state_machine.handlers.delivery.waiting_for_delivery_address_collection_handler import (
    WaitingForDeliveryAddressCollectionHandler,
)
from app.state_machine.handlers.delivery.waiting_for_delivery_eligibility_handler import (
    WaitingForDeliveryEligibilityHandler,
)
from app.state_machine.handlers.order.waiting_for_order_type_handler import (
    WaitingForOrderTypeHandler,
)
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState


class DummyCart:
    def is_empty(self):
        return True


class DummySession:
    def __init__(self):
        self.cart = DummyCart()


class DummyCartSummaryBuilder:
    def build(self, cart):
        return {}


class DummyCheckoutService:
    pass


def test_looks_like_ordering_request_detects_item_slots():
    context = ConversationContext()
    context.last_slots = (SlotValue(name="ITEM", value="Chicken Burger"),)

    assert looks_like_ordering_request(context, "chicken burger") is True


def test_waiting_for_order_type_redirects_unknown_item_like_request():
    context = ConversationContext()
    context.last_slots = (SlotValue(name="ITEM", value="Chicken Burger"),)

    result = WaitingForOrderTypeHandler().handle(
        intent=Intent.UNKNOWN,
        context=context,
        user_text="chicken burger",
        session=None,
    )

    assert result.next_state == ConversationState.WAITING_FOR_ORDER_TYPE
    assert result.response_key == "ordering_blocked_need_order_type"


def test_waiting_for_delivery_eligibility_redirects_unknown_item_like_request():
    context = ConversationContext()
    context.current_prompt_field = "delivery_area"

    result = WaitingForDeliveryEligibilityHandler().handle(
        intent=Intent.UNKNOWN,
        context=context,
        user_text="i want chicken burger",
        session=None,
    )

    assert result.next_state == ConversationState.WAITING_FOR_DELIVERY_ELIGIBILITY
    assert result.response_key == "ordering_blocked_need_delivery_info"
    assert result.response_payload == {"step": "delivery_area"}


def test_waiting_for_delivery_eligibility_accepts_area_text_despite_noisy_item_slot():
    context = ConversationContext()
    context.current_prompt_field = "delivery_area"
    context.last_slots = (SlotValue(name="ITEM", value="washington dc"),)

    result = WaitingForDeliveryEligibilityHandler().handle(
        intent=Intent.UNKNOWN,
        context=context,
        user_text="washington dc",
        session=None,
    )

    assert result.next_state == ConversationState.WAITING_FOR_DELIVERY_ELIGIBILITY
    assert result.response_key == "ask_for_delivery_zip"
    assert context.delivery_address.area == "washington dc"
    assert context.current_prompt_field == "delivery_postal_code"


def test_waiting_for_delivery_address_collection_redirects_unknown_item_like_request():
    context = ConversationContext()
    context.current_prompt_field = "delivery_street"

    handler = WaitingForDeliveryAddressCollectionHandler(
        cart_summary_builder=DummyCartSummaryBuilder(),
        checkout_service=DummyCheckoutService(),
    )
    result = handler.handle(
        intent=Intent.UNKNOWN,
        context=context,
        user_text="i want chicken burger",
        session=DummySession(),
    )

    assert result.next_state == ConversationState.WAITING_FOR_DELIVERY_ADDRESS_COLLECTION
    assert result.response_key == "ordering_blocked_need_delivery_address"
    assert result.response_payload == {"step": "delivery_street"}
