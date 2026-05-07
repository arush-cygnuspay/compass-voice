# D:/Working/Cygnus/compass-voice/tests/api/test_voice_stream_thinness.py
"""Static-source tests guarding the thin-transport rule.

These tests parse ``voice_stream_server.py`` with :mod:`ast` and assert
that the WebSocket handler delegates conversation-layer work to
:class:`ConversationSession` instead of doing it inline.
"""
from __future__ import annotations

import ast
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[2]

VOICE_STREAM_SERVER = APP_ROOT / "app" / "api" / "voice_stream_server.py"
TWILIO_SERVER = APP_ROOT / "app" / "api" / "twilio_server.py"


def _load_module() -> ast.Module:
    return ast.parse(VOICE_STREAM_SERVER.read_text(encoding="utf-8"))


def _find_function(module: ast.Module, name: str) -> ast.AsyncFunctionDef | ast.FunctionDef:
    for node in module.body:
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} not found at module top level")


def _find_inner_function(node: ast.AST, name: str) -> ast.AsyncFunctionDef | ast.FunctionDef:
    for child in ast.walk(node):
        if isinstance(child, (ast.AsyncFunctionDef, ast.FunctionDef)) and child.name == name:
            return child
    raise AssertionError(f"inner function {name!r} not found")


def _walk_calls(node: ast.AST) -> list[ast.Call]:
    return [child for child in ast.walk(node) if isinstance(child, ast.Call)]


def _walk_assigns(node: ast.AST) -> list[ast.Assign]:
    return [child for child in ast.walk(node) if isinstance(child, ast.Assign)]


def test_imports_conversation_session() -> None:
    module = _load_module()
    imports: list[str] = []
    for node in ast.walk(module):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                imports.append(f"{node.module}.{alias.name}")
    assert "app.realtime.conversation_session.ConversationSession" in imports


def test_does_not_import_is_actionable_barge_in_directly() -> None:
    """Barge-in policy decisions live in ConversationSession, not transport."""
    module = _load_module()
    for node in ast.walk(module):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                assert alias.name != "is_actionable_barge_in", (
                    "voice_stream_server should not import is_actionable_barge_in; "
                    "the decision belongs to ConversationSession."
                )


def test_ws_handler_does_not_mutate_session_state() -> None:
    module = _load_module()
    handler = _find_function(module, "twilio_media_ws")

    for assign in _walk_assigns(handler):
        for target in assign.targets:
            if isinstance(target, ast.Attribute) and target.attr == "conversation_state":
                raise AssertionError(
                    "WebSocket handler must not write to .conversation_state directly; "
                    "TurnEngine is the sole writer."
                )


def test_ws_handler_does_not_call_process_turn() -> None:
    module = _load_module()
    handler = _find_function(module, "twilio_media_ws")

    for call in _walk_calls(handler):
        func = call.func
        if isinstance(func, ast.Attribute) and func.attr == "process_turn":
            raise AssertionError(
                "WebSocket handler must not call engine.process_turn directly; "
                "go through ConversationSession.process_committed_turn."
            )


def test_ws_handler_does_not_call_save_session() -> None:
    module = _load_module()
    handler = _find_function(module, "twilio_media_ws")

    for call in _walk_calls(handler):
        func = call.func
        name = (
            func.attr
            if isinstance(func, ast.Attribute)
            else (func.id if isinstance(func, ast.Name) else None)
        )
        if name == "save_session":
            raise AssertionError(
                "WebSocket handler must not persist sessions directly; "
                "ConversationSession owns save coordination."
            )


def test_ws_handler_does_not_call_load_session() -> None:
    module = _load_module()
    handler = _find_function(module, "twilio_media_ws")

    for call in _walk_calls(handler):
        func = call.func
        name = (
            func.attr
            if isinstance(func, ast.Attribute)
            else (func.id if isinstance(func, ast.Name) else None)
        )
        if name == "load_session":
            raise AssertionError(
                "WebSocket handler must not load sessions directly; "
                "ConversationSession owns load coordination."
            )


def test_transcript_callbacks_delegate_committed_text_to_conversation_session() -> None:
    module = _load_module()
    handler = _find_function(module, "twilio_media_ws")

    transcript_cb = _find_inner_function(handler, "on_dg_transcript")
    event_cb = _find_inner_function(handler, "on_dg_event")

    def _has_process_committed_turn(node: ast.AST) -> bool:
        for call in _walk_calls(node):
            func = call.func
            if isinstance(func, ast.Attribute) and func.attr == "process_committed_turn":
                return True
        return False

    assert _has_process_committed_turn(transcript_cb), (
        "Deepgram transcript callback must delegate committed text through "
        "ConversationSession.process_committed_turn."
    )
    assert _has_process_committed_turn(event_cb), (
        "Deepgram event callback must delegate utterance-end commits through "
        "ConversationSession.process_committed_turn."
    )


def test_ws_handler_does_not_define_payment_auto_check_logic() -> None:
    """No nested function inside the WS handler may define payment scheduling."""
    module = _load_module()
    handler = _find_function(module, "twilio_media_ws")

    for inner in ast.walk(handler):
        if isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef)):
            assert inner.name != "_schedule_payment_auto_check", (
                "Payment auto-check scheduling moved to ConversationSession."
            )
            assert inner.name != "process_committed_turn", (
                "process_committed_turn must live on ConversationSession."
            )
            assert inner.name != "_build_response_texts", (
                "Response-text construction moved to "
                "conversation_session.build_response_texts."
            )


def test_ws_handler_does_not_call_schedule_payment_auto_check() -> None:
    module = _load_module()
    handler = _find_function(module, "twilio_media_ws")

    for call in _walk_calls(handler):
        func = call.func
        if isinstance(func, ast.Attribute) and func.attr == "schedule_payment_auto_check":
            raise AssertionError(
                "WebSocket handler must not schedule payment auto-checks directly; "
                "ConversationSession owns that path."
            )


def test_ws_handler_instantiates_conversation_session() -> None:
    module = _load_module()
    handler = _find_function(module, "twilio_media_ws")

    found = False
    for call in _walk_calls(handler):
        func = call.func
        if isinstance(func, ast.Name) and func.id == "ConversationSession":
            found = True
            break
    assert found, "WebSocket handler must construct ConversationSession."


def test_payment_auto_check_constants_only_in_conversation_session() -> None:
    """The PAYMENT_LINK_SENT_KEYS / delay constants moved out of transport."""
    text = VOICE_STREAM_SERVER.read_text(encoding="utf-8")
    assert "PAYMENT_LINK_SENT_KEYS" not in text
    assert "PAYMENT_AUTO_CHECK_DELAY_SECONDS" not in text
    assert "_PAYMENT_AWAITING_STATES" not in text


def test_ws_handler_owns_playback_generation_token() -> None:
    """Cancellation token (playback_generation) must remain in transport scope."""
    module = _load_module()
    handler = _find_function(module, "twilio_media_ws")

    found = False
    for assign in _walk_assigns(handler):
        for target in assign.targets:
            if isinstance(target, ast.Name) and target.id == "playback_generation":
                found = True
                break
        if found:
            break

    assert found, (
        "playback_generation cancellation token must be initialized inside "
        "the WebSocket handler so transport retains ownership."
    )


# ── Transport prompt thinness guards ─────────────────────────────────────────

_HARDCODED_STARTUP_PROMPT = "Welcome to Compass. Is this for pickup or delivery?"

_DEAD_CODE_SYMBOLS = [
    "WaitingForCallerDeviceTypeHandler",
    "WAITING_FOR_CALLER_DEVICE_TYPE",
    "WAITING_FOR_LANDLINE_PICKUP_CONFIRMATION",
    "ask_for_caller_device_type",
    "repeat_caller_device_type",
    "confirm_landline_pickup_only",
    "waiting_for_caller_device_type_handler",
]


def test_voice_stream_server_no_hardcoded_startup_prompt() -> None:
    """Transport must not own the order-type prompt text."""
    text = VOICE_STREAM_SERVER.read_text(encoding="utf-8")
    assert _HARDCODED_STARTUP_PROMPT not in text, (
        f"voice_stream_server.py must not hardcode {_HARDCODED_STARTUP_PROMPT!r}; "
        "use app.state.responder.build('ask_for_order_type', ...) instead."
    )


def test_twilio_server_no_hardcoded_startup_prompt() -> None:
    """Twilio transport must not own the order-type prompt text."""
    text = TWILIO_SERVER.read_text(encoding="utf-8")
    assert _HARDCODED_STARTUP_PROMPT not in text, (
        f"twilio_server.py must not hardcode {_HARDCODED_STARTUP_PROMPT!r}; "
        "use responder.build('ask_for_order_type', ...) instead."
    )


def test_no_dead_caller_device_symbols_in_voice_stream_server() -> None:
    """Dead caller-device symbols must not appear in transport."""
    text = VOICE_STREAM_SERVER.read_text(encoding="utf-8")
    for symbol in _DEAD_CODE_SYMBOLS:
        assert symbol not in text, (
            f"voice_stream_server.py still references dead symbol {symbol!r}."
        )


def test_no_dead_caller_device_symbols_in_twilio_server() -> None:
    """Dead caller-device symbols must not appear in Twilio transport."""
    text = TWILIO_SERVER.read_text(encoding="utf-8")
    for symbol in _DEAD_CODE_SYMBOLS:
        assert symbol not in text, (
            f"twilio_server.py still references dead symbol {symbol!r}."
        )
