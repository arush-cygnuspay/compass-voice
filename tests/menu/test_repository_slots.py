from pathlib import Path

from app.menu.query_result import MenuQueryType
from app.menu.repository import MenuRepository
from app.menu.store import MenuStore
from app.nlu.nlu_result import SlotValue


def _build_repo() -> MenuRepository:
    data_root = Path(__file__).resolve().parents[2] / "app" / "data" / "restaurants" / "steves_grill"
    store = MenuStore(
        menu_path=data_root / "menu.json",
        entity_index_path=data_root / "entity_index.json",
    )
    return MenuRepository(store)


def test_resolve_menu_query_from_slots_uses_first_resolvable_item_slot():
    repo = _build_repo()

    result = repo.resolve_menu_query_from_slots_normalized(
        normalized_user_text="not a thing chicken burger coke",
        slots=(
            SlotValue(name="ITEM", value="not a thing"),
            SlotValue(name="ITEM", value="Chicken Burger"),
            SlotValue(name="ITEM", value="Coke"),
        ),
        fallback_to_text=False,
    )

    assert result.type == MenuQueryType.ITEM
    assert result.item is not None
    assert result.item.name == "Chicken Burger"
