# tests/nlu/test_intent_metadata.py
import pytest

from app.nlu.intent_resolution.intent import Intent
from app.nlu.intent_resolution.intent_metadata import (
    INTENT_REGISTRY,
    IntentMeta,
    IntentMetadataRegistry,
)


# ── Registry defaults ─────────────────────────────────────────────────────────

class TestRegistryDefaults:
    def test_unknown_intent_returns_all_false(self) -> None:
        meta = INTENT_REGISTRY.get(Intent.UNKNOWN)
        assert meta.readonly is False
        assert meta.blocks_mid_item is False
        assert meta.escalates is False
        assert meta.default_handler is None

    def test_greeting_intent_returns_defaults(self) -> None:
        meta = INTENT_REGISTRY.get(Intent.GREETING)
        assert meta == IntentMeta()

    def test_deny_intent_returns_defaults(self) -> None:
        meta = INTENT_REGISTRY.get(Intent.DENY)
        assert meta == IntentMeta()


# ── Read-only interrupt intents ───────────────────────────────────────────────

class TestReadonlyIntents:
    @pytest.mark.parametrize("intent", [
        Intent.ASK_PRICE,
        Intent.SHOW_CART,
        Intent.SHOW_TOTAL,
        Intent.AVAILABILITY_QUERY,
    ])
    def test_readonly_flag(self, intent: Intent) -> None:
        assert INTENT_REGISTRY.get(intent).readonly is True

    def test_ask_price_handler(self) -> None:
        assert INTENT_REGISTRY.get(Intent.ASK_PRICE).default_handler == "ask_price_handler"

    def test_show_cart_handler(self) -> None:
        assert INTENT_REGISTRY.get(Intent.SHOW_CART).default_handler == "cart_handler"

    def test_show_total_handler(self) -> None:
        assert INTENT_REGISTRY.get(Intent.SHOW_TOTAL).default_handler == "cart_handler"

    def test_availability_query_handler(self) -> None:
        assert INTENT_REGISTRY.get(Intent.AVAILABILITY_QUERY).default_handler == "ask_menu_info_handler"

    @pytest.mark.parametrize("intent", [
        Intent.ASK_PRICE,
        Intent.SHOW_CART,
        Intent.SHOW_TOTAL,
        Intent.AVAILABILITY_QUERY,
    ])
    def test_readonly_does_not_block_mid_item(self, intent: Intent) -> None:
        assert INTENT_REGISTRY.get(intent).blocks_mid_item is False


# ── Menu info intents (default_handler but not readonly) ─────────────────────

class TestMenuInfoIntents:
    @pytest.mark.parametrize("intent", [
        Intent.ASK_ITEM_INFO,
        Intent.ASK_MENU_INFO,
        Intent.ASK_OPTIONS,
        Intent.BROWSE_MENU,
        Intent.BROWSE_CATEGORY,
        Intent.RECOMMENDATION_QUERY,
        Intent.SHOW_MENU,
    ])
    def test_menu_info_handler(self, intent: Intent) -> None:
        meta = INTENT_REGISTRY.get(intent)
        assert meta.default_handler == "ask_menu_info_handler"
        assert meta.readonly is False
        assert meta.blocks_mid_item is False


# ── Mid-item blockers ─────────────────────────────────────────────────────────

class TestMidItemBlockers:
    @pytest.mark.parametrize("intent", [
        Intent.START_ORDER,
        Intent.END_ADDING,
        Intent.CHECKOUT,
        Intent.CONFIRM_ORDER,
        Intent.FINISH_ORDER,
        Intent.PAYMENT_REQUEST,
        Intent.REVIEW_ORDER,
    ])
    def test_blocks_mid_item_flag(self, intent: Intent) -> None:
        assert INTENT_REGISTRY.get(intent).blocks_mid_item is True

    @pytest.mark.parametrize("intent", [
        Intent.START_ORDER,
        Intent.END_ADDING,
        Intent.CHECKOUT,
        Intent.CONFIRM_ORDER,
        Intent.FINISH_ORDER,
        Intent.PAYMENT_REQUEST,
        Intent.REVIEW_ORDER,
    ])
    def test_blockers_are_not_readonly(self, intent: Intent) -> None:
        assert INTENT_REGISTRY.get(intent).readonly is False

    @pytest.mark.parametrize("intent", [
        Intent.START_ORDER,
        Intent.END_ADDING,
        Intent.CHECKOUT,
        Intent.CONFIRM_ORDER,
        Intent.FINISH_ORDER,
        Intent.PAYMENT_REQUEST,
        Intent.REVIEW_ORDER,
    ])
    def test_blockers_have_no_default_handler(self, intent: Intent) -> None:
        assert INTENT_REGISTRY.get(intent).default_handler is None


# ── Escalation ────────────────────────────────────────────────────────────────

class TestEscalation:
    def test_request_agent_escalates(self) -> None:
        assert INTENT_REGISTRY.get(Intent.REQUEST_AGENT).escalates is True

    def test_request_agent_is_not_readonly(self) -> None:
        assert INTENT_REGISTRY.get(Intent.REQUEST_AGENT).readonly is False

    def test_request_agent_does_not_block_mid_item(self) -> None:
        assert INTENT_REGISTRY.get(Intent.REQUEST_AGENT).blocks_mid_item is False

    def test_no_other_intent_escalates(self) -> None:
        escalating = [i for i in Intent if INTENT_REGISTRY.get(i).escalates]
        assert escalating == [Intent.REQUEST_AGENT]


# ── IntentMeta dataclass ──────────────────────────────────────────────────────

class TestIntentMeta:
    def test_defaults(self) -> None:
        meta = IntentMeta()
        assert meta.readonly is False
        assert meta.blocks_mid_item is False
        assert meta.escalates is False
        assert meta.default_handler is None

    def test_frozen(self) -> None:
        meta = IntentMeta(readonly=True)
        with pytest.raises(Exception):
            meta.readonly = False  # type: ignore[misc]

    def test_equality(self) -> None:
        assert IntentMeta() == IntentMeta()
        assert IntentMeta(readonly=True) != IntentMeta()


# ── Custom registry ───────────────────────────────────────────────────────────

class TestCustomRegistry:
    def test_custom_entry_overrides_default(self) -> None:
        reg = IntentMetadataRegistry({Intent.GREETING: IntentMeta(readonly=True)})
        assert reg.get(Intent.GREETING).readonly is True

    def test_missing_intent_returns_default(self) -> None:
        reg = IntentMetadataRegistry({})
        assert reg.get(Intent.ADD_ITEM) == IntentMeta()
