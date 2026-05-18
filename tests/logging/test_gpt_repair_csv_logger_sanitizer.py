# tests/logging/test_gpt_repair_csv_logger_sanitizer.py
"""Tests for the Part 7 sanitizer additions in gpt_repair_csv_logger.

Covers:
- ISO timestamp preservation in sanitize_string()
- Phone / email / payment-link redaction in user text
- _sanitize_json_field(): JSON-aware sanitization that keeps non-PII values
- _sanitize_row(): enum/log field skip list + _json suffix handling
- sanitize_record(): field-name-aware dict traversal
"""
from __future__ import annotations

import json

import pytest

from app.logging.gpt_repair_csv_logger import (
    _SANITIZE_SKIP_FIELDS,
    _sanitize_json_field,
    _sanitize_row,
    sanitize_record,
    sanitize_string,
)


# ---------------------------------------------------------------------------
# ISO timestamp preservation
# ---------------------------------------------------------------------------


class TestIsoTimestampPreservation:
    def test_iso_timestamp_z_suffix_unchanged(self):
        ts = "2024-01-15T12:34:56Z"
        assert sanitize_string(ts) == ts

    def test_iso_timestamp_with_microseconds_unchanged(self):
        ts = "2026-05-18T10:30:00.123456+00:00"
        assert sanitize_string(ts) == ts

    def test_iso_timestamp_with_ms_and_z_unchanged(self):
        ts = "2099-12-31T23:59:59.999Z"
        assert sanitize_string(ts) == ts

    def test_plain_date_no_t_still_checked(self):
        # YYYY-MM-DD without T has no _ISO_TIMESTAMP_RE match,
        # so phone regex runs — but ISO dates (4-2-2 digits) don't match _PHONE_RE
        date_only = "2024-01-15"
        # Should not be mangled (not 7+ consecutive digits)
        assert sanitize_string(date_only) == date_only

    def test_iso_timestamp_email_still_redacted(self):
        # Even in an ISO timestamp string, embedded email should be stripped
        text = "2024-01-15T12:34:56Z user@example.com follow-up"
        result = sanitize_string(text)
        assert "user@example.com" not in result
        assert "[REDACTED_EMAIL]" in result
        # The timestamp prefix itself is preserved
        assert "2024-01-15T" in result


# ---------------------------------------------------------------------------
# Phone / email / payment-link redaction
# ---------------------------------------------------------------------------


class TestPiiRedactionInUserText:
    def test_phone_ten_digits_redacted(self):
        result = sanitize_string("call me at 5551234567 thanks")
        assert "5551234567" not in result
        assert "[REDACTED_PHONE]" in result

    def test_phone_formatted_dashes_redacted(self):
        result = sanitize_string("my number is 555-867-5309")
        assert "555-867-5309" not in result
        assert "[REDACTED_PHONE]" in result

    def test_phone_formatted_spaces_redacted(self):
        result = sanitize_string("call 555 867 5309 please")
        assert "555 867 5309" not in result
        assert "[REDACTED_PHONE]" in result

    def test_email_redacted(self):
        result = sanitize_string("send to user@domain.com now")
        assert "user@domain.com" not in result
        assert "[REDACTED_EMAIL]" in result

    def test_payment_link_redacted(self):
        result = sanitize_string("pay at https://pay.stripe.com/checkout/abc123")
        assert "pay.stripe.com" not in result
        assert "[REDACTED_URL]" in result

    def test_checkout_link_redacted(self):
        result = sanitize_string("go to https://example.com/checkout/order/42")
        assert "example.com/checkout" not in result
        assert "[REDACTED_URL]" in result

    def test_clean_text_unchanged(self):
        text = "I want a burger please"
        assert sanitize_string(text) == text

    def test_six_digits_not_redacted(self):
        # 6-digit sequences are NOT phone numbers — must not be redacted
        result = sanitize_string("code 123456 is valid")
        assert "123456" in result


# ---------------------------------------------------------------------------
# _sanitize_json_field — JSON-aware sanitization
# ---------------------------------------------------------------------------


class TestSanitizeJsonField:
    def test_text_leaf_with_phone_sanitized(self):
        raw = json.dumps({"user_text": "call 5551234567 now"})
        result = _sanitize_json_field(raw)
        data = json.loads(result)
        assert "5551234567" not in data["user_text"]
        assert "[REDACTED_PHONE]" in data["user_text"]

    def test_text_leaf_with_email_sanitized(self):
        raw = json.dumps({"text": "reach me at admin@example.com"})
        result = _sanitize_json_field(raw)
        data = json.loads(result)
        assert "admin@example.com" not in data["text"]
        assert "[REDACTED_EMAIL]" in data["text"]

    def test_numeric_value_preserved(self):
        raw = json.dumps({"confidence": 0.9372, "count": 42})
        result = _sanitize_json_field(raw)
        data = json.loads(result)
        assert data["confidence"] == pytest.approx(0.9372)
        assert data["count"] == 42

    def test_bool_value_preserved(self):
        raw = json.dumps({"applied": False, "eligible": True})
        result = _sanitize_json_field(raw)
        data = json.loads(result)
        assert data["applied"] is False
        assert data["eligible"] is True

    def test_null_value_preserved(self):
        raw = json.dumps({"size": None, "variant": None})
        result = _sanitize_json_field(raw)
        data = json.loads(result)
        assert data["size"] is None
        assert data["variant"] is None

    def test_nested_dict_text_sanitized(self):
        payload = {"outer": {"inner_text": "email user@test.com here"}}
        raw = json.dumps(payload)
        result = _sanitize_json_field(raw)
        data = json.loads(result)
        assert "user@test.com" not in data["outer"]["inner_text"]

    def test_array_of_strings_sanitized(self):
        raw = json.dumps({"notes": ["clean text", "call 5551234567"]})
        result = _sanitize_json_field(raw)
        data = json.loads(result)
        assert data["notes"][0] == "clean text"
        assert "5551234567" not in data["notes"][1]

    def test_invalid_json_falls_back_gracefully(self):
        # Not valid JSON — should sanitize as raw string without crashing
        malformed = '{"key": "value", broken'
        result = _sanitize_json_field(malformed)
        assert isinstance(result, str)  # must return a string

    def test_invalid_json_with_pii_still_redacted(self):
        malformed = '{"key": broken, email: user@test.com'
        result = _sanitize_json_field(malformed)
        assert "user@test.com" not in result

    def test_output_is_valid_json(self):
        raw = json.dumps({"text": "hi", "num": 1, "flag": True, "nil": None})
        result = _sanitize_json_field(raw)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_iso_timestamp_in_json_preserved(self):
        raw = json.dumps({"ts": "2024-01-15T12:34:56Z", "text": "hello"})
        result = _sanitize_json_field(raw)
        data = json.loads(result)
        assert data["ts"] == "2024-01-15T12:34:56Z"


# ---------------------------------------------------------------------------
# _sanitize_row — enum/log skip list + _json suffix
# ---------------------------------------------------------------------------


class TestSanitizeRow:
    def test_gpt_decision_preserved(self):
        row = {"gpt_decision": "no_repair", "user_text": "I want a burger"}
        result = _sanitize_row(row)
        assert result["gpt_decision"] == "no_repair"

    def test_state_before_preserved(self):
        row = {"state_before": "waiting_for_modifier"}
        result = _sanitize_row(row)
        assert result["state_before"] == "waiting_for_modifier"

    def test_local_intent_preserved(self):
        row = {"local_intent": "ORDER_ITEM"}
        result = _sanitize_row(row)
        assert result["local_intent"] == "ORDER_ITEM"

    def test_gpt_apply_reason_preserved(self):
        row = {"gpt_apply_reason": "shadow_mode"}
        result = _sanitize_row(row)
        assert result["gpt_apply_reason"] == "shadow_mode"

    def test_gpt_skipped_reason_preserved(self):
        row = {"gpt_skipped_reason": "daily_budget_exceeded"}
        result = _sanitize_row(row)
        assert result["gpt_skipped_reason"] == "daily_budget_exceeded"

    def test_gpt_fallback_type_preserved(self):
        row = {"gpt_fallback_type": "hard_fallback"}
        result = _sanitize_row(row)
        assert result["gpt_fallback_type"] == "hard_fallback"

    def test_all_skip_fields_preserved(self):
        # All fields in _SANITIZE_SKIP_FIELDS must pass through unchanged
        value = "some_enum_value"
        for field in _SANITIZE_SKIP_FIELDS:
            row = {field: value}
            result = _sanitize_row(row)
            assert result[field] == value, f"Field '{field}' was incorrectly sanitized"

    def test_user_text_phone_redacted(self):
        row = {"user_text": "my number is 5551234567"}
        result = _sanitize_row(row)
        assert "5551234567" not in result["user_text"]
        assert "[REDACTED_PHONE]" in result["user_text"]

    def test_user_text_email_redacted(self):
        row = {"user_text": "email me at test@example.com"}
        result = _sanitize_row(row)
        assert "test@example.com" not in result["user_text"]

    def test_json_field_uses_json_aware_path(self):
        payload = {"slots": {"phone": "5551234567"}, "count": 3}
        row = {"local_slots_json": json.dumps(payload)}
        result = _sanitize_row(row)
        data = json.loads(result["local_slots_json"])
        assert "5551234567" not in data["slots"]["phone"]
        # Numeric preserved
        assert data["count"] == 3

    def test_non_string_values_pass_through(self):
        row = {"gpt_called": True, "turn_index": 5, "gpt_total_ms": 123.4}
        result = _sanitize_row(row)
        assert result["gpt_called"] is True
        assert result["turn_index"] == 5
        assert result["gpt_total_ms"] == pytest.approx(123.4)

    def test_gpt_eligible_reason_preserved(self):
        row = {"gpt_eligible_reason": "unknown_with_context"}
        result = _sanitize_row(row)
        assert result["gpt_eligible_reason"] == "unknown_with_context"

    def test_gpt_model_preserved(self):
        row = {"gpt_model": "gpt-4o-mini"}
        result = _sanitize_row(row)
        assert result["gpt_model"] == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# sanitize_record — field-name-aware dict traversal
# ---------------------------------------------------------------------------


class TestSanitizeRecordFieldAware:
    def test_skip_field_in_dict_preserved(self):
        record = {"gpt_decision": "repair", "user_text": "hello 5551234567"}
        result = sanitize_record(record)
        assert result["gpt_decision"] == "repair"
        assert "5551234567" not in result["user_text"]

    def test_json_suffix_field_uses_json_path(self):
        payload = {"text": "call 5551234567 now", "num": 99}
        record = {"local_slots_json": json.dumps(payload)}
        result = sanitize_record(record)
        data = json.loads(result["local_slots_json"])
        assert "5551234567" not in data["text"]
        assert data["num"] == 99

    def test_nested_dict_traversed(self):
        record = {
            "outer": {
                "gpt_decision": "no_repair",
                "inner_text": "email me@foo.com",
            }
        }
        result = sanitize_record(record)
        # gpt_decision is a skip field — but only at top-level dict key
        # When nested, it depends on recursive call passing field name
        # The inner dict's keys should be treated the same way
        # gpt_decision key at inner level is still in SKIP list
        assert result["outer"]["gpt_decision"] == "no_repair"
        assert "me@foo.com" not in result["outer"]["inner_text"]

    def test_list_of_strings_sanitized(self):
        record = {"items": ["clean text", "call 5551234567"]}
        result = sanitize_record(record)
        assert result["items"][0] == "clean text"
        assert "5551234567" not in result["items"][1]

    def test_non_string_scalars_unchanged(self):
        record = {"count": 5, "flag": True, "nothing": None}
        result = sanitize_record(record)
        assert result["count"] == 5
        assert result["flag"] is True
        assert result["nothing"] is None
