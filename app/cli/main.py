# app/cli/main.py

from pathlib import Path
import uuid

from app.core.turn_engine import TurnEngine
from app.core.response_builder import ResponseBuilder
from app.menu.repository import MenuRepository
from app.menu.store import MenuStore
from app.menu.exceptions import MenuLoadError
from app.nlu.intent_resolution.intent import Intent
from app.state_machine.state_router import StateRouter
from app.session.repository import load_session, save_session


def load_menu_store(restaurant_id: str) -> MenuStore:
    project_root = Path(__file__).resolve().parents[2]
    data_root = project_root / "app" / "data" / "restaurants" / restaurant_id

    menu_path = data_root / "menu.json"
    entity_index_path = data_root / "entity_index.json"

    return MenuStore(menu_path, entity_index_path)


def main():
    print("=== Compass Voice (CLI Mode) ===")
    print("Type 'exit' to quit.\n")

    # ----------------------------
    # Session identity
    # ----------------------------
    session_id = "cli-" + str(uuid.uuid4())
    restaurant_id = "demo"

    # ----------------------------
    # Load menu
    # ----------------------------
    try:
        store = load_menu_store(restaurant_id)
    except MenuLoadError as e:
        print("❌ Failed to load menu:", e)
        return

    menu_repo = MenuRepository(store)

    # ----------------------------
    # Core components
    # ----------------------------
    router = StateRouter()
    engine = TurnEngine(router, menu_repo)
    responder = ResponseBuilder(menu_repo)

    # ----------------------------
    # Load session from Redis
    # ----------------------------
    session = load_session(session_id, restaurant_id)

    # ----------------------------
    # Main loop
    # ----------------------------
    while True:
        user_input = input("\nYOU: ").strip()

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        response_key = engine.process_turn(
            session=session,
            user_text=user_input,
        )

        # Persist session after each turn
        save_session(session)

        reply = responder.build(response_key, session.conversation_context)

        print(f"\nBOT: {reply}")


if __name__ == "__main__":
    main()
