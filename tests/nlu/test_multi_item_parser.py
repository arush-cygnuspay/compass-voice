import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.menu.store import MenuStore
from app.nlu.multi_item_parser import parse_multi_item_utterance
from app.nlu.nlu_result import SlotValue
from app.nlu.query_normalization.text_preprocessor import normalize_text


def _build_demo_store() -> MenuStore:
    data_root = Path(__file__).resolve().parents[2] / "app" / "data" / "restaurants" / "demo"
    return MenuStore(
        menu_path=data_root / "menu.json",
        entity_index_path=data_root / "entity_index.json",
    )


def _slot(name: str, value: str, text: str) -> SlotValue:
    start = text.index(value)
    end = start + len(value)
    return SlotValue(name=name, value=value, raw=value, start=start, end=end, confidence=1.0)


def test_parse_multi_item_utterance_does_not_split_attached_with_items():
    text = normalize_text(
        "a chicken burger with coke and extra cheese red onions and no sauce and two chicken tacos with coke and american cheese"
    )

    slots = (
        _slot("ITEM", "chicken burger", text),
        _slot("ITEM", "coke", text),
        _slot("ITEM", "chicken tacos", text),
        SlotValue(
            name="ITEM",
            value="coke",
            raw="coke",
            start=text.rindex("coke"),
            end=text.rindex("coke") + len("coke"),
            confidence=1.0,
        ),
    )

    segments = parse_multi_item_utterance(text, slots)

    assert len(segments) == 2
    assert segments[0].item_slot_value == "chicken burger"
    assert segments[1].item_slot_value == "chicken tacos"
    assert "with coke and extra cheese red onions and no sauce" in segments[0].raw_text
    assert "with coke and american cheese" in segments[1].raw_text


def test_parse_multi_item_utterance_uses_menu_truth_for_boundary_items():
    store = _build_demo_store()
    text = normalize_text(
        "a chicken taco with a coke and american cheese plus jelly and sausage and also a chicken burger with red onions"
    )

    slots = (
        _slot("ITEM", "chicken taco", text),
        SlotValue(name="ITEM", value="Coke (12 oz.)", raw="coke", confidence=1.0),
        _slot("ITEM", "american cheese", text),
        SlotValue(name="MODIFIER", value="Jelly", raw="jelly", start=text.index("jelly"), end=text.index("jelly") + len("jelly"), confidence=1.0),
        SlotValue(name="MODIFIER", value="Sausage", raw="sausage", start=text.index("sausage"), end=text.index("sausage") + len("sausage"), confidence=1.0),
        _slot("ITEM", "chicken burger", text),
        SlotValue(name="MODIFIER", value="Red Onions", raw="red onions", start=text.index("red onions"), end=text.index("red onions") + len("red onions"), confidence=1.0),
    )

    segments = parse_multi_item_utterance(text, slots, menu_store=store)

    assert len(segments) == 2
    assert segments[0].item_slot_value == "chicken taco"
    assert segments[1].item_slot_value == "chicken burger"
    assert "american cheese plus jelly and sausage" in segments[0].raw_text
    assert "with red onions" in segments[1].raw_text

    first_segment_values = {
        normalize_text(str(slot.value))
        for slot in segments[0].slots
        if isinstance(slot.value, str)
    }
    assert "american cheese" in first_segment_values
    assert "jelly" in first_segment_values
    assert "sausage" in first_segment_values
    assert "chicken burger" not in first_segment_values
