"""Tests for payment + skip control-intent routing through the central registry."""
import logging
import unittest

from app.nlu.intent_resolution.intent import Intent
from app.state_machine.control_intent_resolver import (
    ControlIntentKind,
    resolve_control_intent,
)
from app.state_machine.models.conversation_state import ConversationState


class _EventCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.events: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.events.append(record)


class PaymentIntentRoutingTests(unittest.TestCase):
    """Phrase fallback / payment_state_phrase precedence tests for the
    new ControlIntentKind variants and the SKIP kind."""

    def setUp(self) -> None:
        self.capture = _EventCapture()
        self.logger = logging.getLogger("app.state_machine.control_intent_resolver")
        self.previous_level = self.logger.level
        self.logger.addHandler(self.capture)
        self.logger.setLevel(logging.INFO)

    def tearDown(self) -> None:
        self.logger.removeHandler(self.capture)
        self.logger.setLevel(self.previous_level)

    def _resolve(
        self,
        text: str,
        state: ConversationState,
        intent: Intent | str | None = Intent.UNKNOWN,
    ):
        return resolve_control_intent(
            text,
            intent,
            None,
            state,
            None,
            nlu_result=None,
            intent_confidence=0.0,
        )

    # ── Payment-state phrase fallback ───────────────────────────────

    def test_stay_on_call_resolves_in_waiting_for_payment(self):
        result = self._resolve("stay on the line", ConversationState.WAITING_FOR_PAYMENT)
        self.assertIsNotNone(result)
        self.assertEqual(result.kind, ControlIntentKind.PAYMENT_STAY_ON_CALL)
        self.assertEqual(result.source, "payment_state_phrase")

    def test_stay_on_call_resolves_in_waiting_for_checkout_completion(self):
        result = self._resolve(
            "stay on the line", ConversationState.WAITING_FOR_CHECKOUT_COMPLETION
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.kind, ControlIntentKind.PAYMENT_STAY_ON_CALL)

    def test_after_call_resolves_in_payment_state(self):
        result = self._resolve("ill do it later", ConversationState.WAITING_FOR_PAYMENT)
        self.assertIsNotNone(result)
        self.assertEqual(result.kind, ControlIntentKind.PAYMENT_AFTER_CALL)

    def test_cannot_open_link_resolves_in_payment_state(self):
        result = self._resolve(
            "i cant open the link", ConversationState.WAITING_FOR_PAYMENT
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.kind, ControlIntentKind.PAYMENT_CANNOT_OPEN_LINK)

    # ── Non-payment states do NOT fire payment kinds ────────────────

    def test_stay_on_call_does_not_fire_in_idle_state(self):
        result = self._resolve("stay on the line", ConversationState.IDLE)
        # Either None or a different kind, but not PAYMENT_STAY_ON_CALL
        if result is not None:
            self.assertNotEqual(result.kind, ControlIntentKind.PAYMENT_STAY_ON_CALL)

    def test_after_call_does_not_fire_in_confirming_order(self):
        result = self._resolve("ill do it later", ConversationState.CONFIRMING_ORDER)
        if result is not None:
            self.assertNotEqual(result.kind, ControlIntentKind.PAYMENT_AFTER_CALL)

    def test_cannot_open_link_does_not_fire_in_idle_state(self):
        result = self._resolve("i cant open the link", ConversationState.IDLE)
        if result is not None:
            self.assertNotEqual(result.kind, ControlIntentKind.PAYMENT_CANNOT_OPEN_LINK)

    # ── Payment-state precedence over generic AFFIRM/DENY ───────────

    def test_after_call_phrase_wins_over_unrelated_nlu_label(self):
        # Even if NLU classified utterance as DENY, the payment-state
        # phrase scan must short-circuit (legacy precedence).
        result = self._resolve(
            "ill do it later",
            ConversationState.WAITING_FOR_PAYMENT,
            intent=Intent.DENY,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.kind, ControlIntentKind.PAYMENT_AFTER_CALL)

    # ── SKIP kind (delivery secondary address) ─────────────────────

    def test_skip_resolves_in_optional_field_state(self):
        result = self._resolve(
            "skip", ConversationState.WAITING_FOR_DELIVERY_ADDRESS_COLLECTION
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.kind, ControlIntentKind.SKIP)

    def test_skip_does_not_fire_outside_optional_field_states(self):
        result = self._resolve("skip", ConversationState.WAITING_FOR_PAYMENT)
        if result is not None:
            self.assertNotEqual(result.kind, ControlIntentKind.SKIP)

    def test_nothing_resolves_to_skip_in_optional_field_state(self):
        result = self._resolve(
            "nothing", ConversationState.WAITING_FOR_DELIVERY_ADDRESS_COLLECTION
        )
        self.assertIsNotNone(result)
        # "nothing" is in both _SKIP_PHRASES and _DENY_PHRASES, but the
        # SKIP check runs first in _OPTIONAL_FIELD_STATES.
        self.assertEqual(result.kind, ControlIntentKind.SKIP)

    # ── Phrase-fallback log event ───────────────────────────────────

    def test_phrase_fallback_used_event_emitted_for_payment_state_phrase(self):
        self._resolve("stay on the line", ConversationState.WAITING_FOR_PAYMENT)
        events = [r for r in self.capture.events if r.getMessage() == "phrase_fallback_used"]
        self.assertEqual(len(events), 1)
        self.assertEqual(getattr(events[0], "kind", None), "payment_stay_on_call")
        self.assertEqual(getattr(events[0], "source", None), "payment_state_phrase")


class PaymentIntentRegistryStateGatingTests(unittest.TestCase):
    """Verify the new registry entries are state-gated."""

    def test_skip_label_resolves_only_in_optional_field_states(self):
        # If NLU were to emit "skip" as the label, the registry rule
        # should fire in WAITING_FOR_DELIVERY_ADDRESS_COLLECTION but
        # not in other states.
        result_in_state = resolve_control_intent(
            "skip",
            "skip",
            None,
            ConversationState.WAITING_FOR_DELIVERY_ADDRESS_COLLECTION,
            None,
            intent_confidence=1.0,
        )
        self.assertIsNotNone(result_in_state)
        self.assertEqual(result_in_state.kind, ControlIntentKind.SKIP)

        result_off_state = resolve_control_intent(
            "skip",
            "skip",
            None,
            ConversationState.IDLE,
            None,
            intent_confidence=1.0,
        )
        if result_off_state is not None:
            self.assertNotEqual(result_off_state.kind, ControlIntentKind.SKIP)


if __name__ == "__main__":
    unittest.main()
