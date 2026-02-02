from pathlib import Path

from app.core.response_builder import ResponseBuilder
from app.core.turn_engine import TurnEngine
from app.menu.exceptions import MenuLoadError
from app.menu.repository import MenuRepository
from app.menu.store import MenuStore
from app.session.session import Session
from app.state_machine.state_router import StateRouter


# =================================================
# CONFIG — PROJECT ROOT SAFE
# =================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "restaurants"

RESTAURANT_ID = "demo"
BASE_PATH = DATA_ROOT / RESTAURANT_ID
MENU_PATH = BASE_PATH / "menu.json"
ENTITY_INDEX_PATH = BASE_PATH / "entity_index.json"


def _pick_item(store: MenuStore):
    """
    Prefer an item that:
    - does NOT require size selection (fixed/unit)
    - has side groups AND modifier groups
    """
    item_with_side = None
    item_with_modifier = None
    item_with_both = None

    for item in store.items.values():
        if item.pricing.mode == "variant":
            continue

        if item.side_groups and item_with_side is None:
            item_with_side = item
        if item.modifier_groups and item_with_modifier is None:
            item_with_modifier = item
        if item.side_groups and item.modifier_groups and item_with_both is None:
            item_with_both = item

    return item_with_both or item_with_side or item_with_modifier


def main():
    print("=== SLOT OPTIONS SMOKE TEST ===\n")

    try:
        store = MenuStore(MENU_PATH, ENTITY_INDEX_PATH)
        repo = MenuRepository(store)
    except MenuLoadError as e:
        print("❌ Failed to load menu:", e)
        raise

    engine = TurnEngine(StateRouter(), repo)
    responder = ResponseBuilder(repo)
    session = Session(session_id="smoke", restaurant_id=RESTAURANT_ID)

    item = _pick_item(store)
    assert item is not None, "No suitable item found in demo menu."

    print("Using item:", item.name)

    # Turn 1: start add flow
    out = engine.process_turn(session, item.name)
    print("T1:", out.response_key, "| state:", session.conversation_state.name, "| intent:", session.last_intent.name)
    print("R1:", responder.build(out.response_key, session.conversation_context, out.response_payload), "\n")

    # Turn 2: ask for options in current slot
    out = engine.process_turn(session, "what are my options")
    print("T2:", out.response_key, "| state:", session.conversation_state.name, "| intent:", session.last_intent.name)
    print("R2:", responder.build(out.response_key, session.conversation_context, out.response_payload), "\n")

    assert out.response_key in {"repeat_side_options", "repeat_modifier_options", "repeat_size_options", "ask_for_size"}, (
        "Expected an option-listing response during slot filling."
    )

    # If we're waiting for a side, select the first side and proceed.
    if session.conversation_state.name == "WAITING_FOR_SIDE":
        cur_item = store.get_item(session.conversation_context.current_item_id)
        group = cur_item.side_groups[session.conversation_context.current_side_group_index]
        choice_name = group.choices[0].name
        out = engine.process_turn(session, choice_name)
        print("T3:", out.response_key, "| state:", session.conversation_state.name, "| intent:", session.last_intent.name)
        print("R3:", responder.build(out.response_key, session.conversation_context, out.response_payload), "\n")

    # If we're waiting for a modifier, ask options again.
    if session.conversation_state.name == "WAITING_FOR_MODIFIER":
        out = engine.process_turn(session, "what's available")
        print("T4:", out.response_key, "| state:", session.conversation_state.name, "| intent:", session.last_intent.name)
        print("R4:", responder.build(out.response_key, session.conversation_context, out.response_payload), "\n")
        assert out.response_key == "repeat_modifier_options", "Expected modifier options to repeat."

    print("SLOT OPTIONS SMOKE TEST PASSED")


if __name__ == "__main__":
    main()

