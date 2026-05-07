# tests/responses/test_progress_prompt_partial_selection.py
"""Phase B regression tests — valid partial-selection must NOT say 'didn't catch'."""
import pytest

from app.responses.item.format_utils import _progress_prompt, _group_payload


# ---------------------------------------------------------------------------
# _progress_prompt unit tests
# ---------------------------------------------------------------------------

class TestProgressPromptPartialSelection:
    """When selected_count > 0 and repeat_reason is not 'invalid',
    _progress_prompt must NOT include the invalid_lead text."""

    INVALID_LEAD = "I didn't catch a valid platter sides option."

    def _payload(self, **overrides) -> dict:
        base = {
            "top_choices": ["Potato Salad", "Corn on the Cob", "Cole Slaw"],
            "all_choices": ["Rice", "Potato Salad", "Corn on the Cob", "Cole Slaw"],
            "selected_count": 1,
            "min_selector": 2,
            "max_selector": 2,
            "remaining_to_min": 1,
            "remaining_to_max": 1,
        }
        base.update(overrides)
        return base

    def test_need_more_reason_no_invalid_lead(self):
        payload = self._payload(repeat_reason="need_more")
        result = _progress_prompt(payload, item_word="side", invalid_lead=self.INVALID_LEAD)
        assert "didn't catch" not in result
        assert "Pick 1 more" in result

    def test_no_reason_valid_partial_no_invalid_lead(self):
        """No repeat_reason set; selected_count > 0 — must not say 'didn't catch'."""
        payload = self._payload()
        result = _progress_prompt(payload, item_word="side", invalid_lead=self.INVALID_LEAD)
        assert "didn't catch" not in result
        assert "Pick 1 more" in result

    def test_invalid_reason_uses_invalid_lead(self):
        """repeat_reason='invalid' means user's input was bad — invalid_lead IS expected."""
        payload = self._payload(selected_count=0, repeat_reason="invalid",
                                remaining_to_min=2, remaining_to_max=2)
        result = _progress_prompt(payload, item_word="side", invalid_lead=self.INVALID_LEAD)
        assert "didn't catch" in result

    def test_zero_selected_no_reason_falls_to_choose(self):
        payload = self._payload(selected_count=0, remaining_to_min=1, remaining_to_max=1)
        result = _progress_prompt(payload, item_word="side", invalid_lead=self.INVALID_LEAD)
        assert "Please choose" in result
        assert "didn't catch" not in result

    def test_options_listed_after_pick_more(self):
        payload = self._payload(repeat_reason="need_more")
        result = _progress_prompt(payload, item_word="side", invalid_lead=self.INVALID_LEAD)
        assert "Potato Salad" in result or "Corn on the Cob" in result or "Cole Slaw" in result

    def test_one_valid_one_invalid_name_acknowledgement(self):
        """Caller sets matched_names + unmatched_names; _build_entity_feedback handles ack.
        _progress_prompt itself should not say 'didn't catch' for the valid part."""
        payload = self._payload(
            repeat_reason="need_more",
            matched_names=["Rice"],
            unmatched_names=["pineapple"],
        )
        result = _progress_prompt(payload, item_word="side", invalid_lead=self.INVALID_LEAD)
        assert "didn't catch" not in result
        assert "Pick 1 more" in result


# ---------------------------------------------------------------------------
# _group_payload propagation tests
# ---------------------------------------------------------------------------

class TestGroupPayloadPropagation:
    """_group_payload must forward control fields from the outer handler payload."""

    def _base_payload(self, **extra) -> dict:
        return {"repeat_reason": "need_more", "matched_names": ["Rice"], **extra}

    def test_repeat_reason_propagated(self):
        result = _group_payload(
            payload=self._base_payload(),
            group_name="Platter Sides",
            option_names=["Rice", "Potato Salad", "Corn on the Cob"],
            selected_names=["Rice"],
            min_selector=2,
            max_selector=2,
        )
        assert result["repeat_reason"] == "need_more"

    def test_matched_names_propagated(self):
        result = _group_payload(
            payload=self._base_payload(),
            group_name="Platter Sides",
            option_names=["Rice", "Potato Salad", "Corn on the Cob"],
            selected_names=["Rice"],
            min_selector=2,
            max_selector=2,
        )
        assert result["matched_names"] == ["Rice"]

    def test_unmatched_names_propagated(self):
        payload = self._base_payload(unmatched_names=["pineapple"])
        result = _group_payload(
            payload=payload,
            group_name="Platter Sides",
            option_names=["Rice", "Potato Salad"],
            selected_names=["Rice"],
            min_selector=2,
            max_selector=2,
        )
        assert result["unmatched_names"] == ["pineapple"]

    def test_control_fields_absent_when_not_in_payload(self):
        result = _group_payload(
            payload={},
            group_name="Platter Sides",
            option_names=["Rice", "Potato Salad"],
            selected_names=[],
            min_selector=2,
            max_selector=2,
        )
        assert "repeat_reason" not in result
        assert "matched_names" not in result
