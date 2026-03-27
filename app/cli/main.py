# app/cli/main.py
from __future__ import annotations

import uuid

from app.bootstrap.runtime import build_runtime
from app.session.repository import load_session, save_session


def main() -> None:
    print("=== Compass Voice (CLI Mode) ===")
    print("Type 'exit' to quit.\n")

    session_id = "cli-" + str(uuid.uuid4())
    restaurant_id = "demo"

    runtime = build_runtime(restaurant_id=restaurant_id)
    session = load_session(session_id, restaurant_id)

    while True:
        user_input = input("\nYOU: ").strip()

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        turn_output = runtime.engine.process_turn(
            session=session,
            user_text=user_input,
        )

        save_session(session)

        reply = runtime.responder.build(
            response_key=turn_output.response_key,
            context=session.conversation_context,
            payload=turn_output.response_payload,
        )

        print(f"\nBOT: {reply}")


if __name__ == "__main__":
    main()