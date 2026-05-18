import unittest
from pathlib import Path

from app.menu.query_result import MenuQueryType
from app.menu.repository import MenuRepository
from app.menu.store import MenuStore


def _build_repo() -> MenuRepository:
    data_root = Path(__file__).resolve().parents[2] / "app" / "data" / "restaurants" / "steves_grill"
    store = MenuStore(
        menu_path=data_root / "menu.json",
        entity_index_path=data_root / "entity_index.json",
    )
    return MenuRepository(store)


class NumberedItemMatchingTests(unittest.TestCase):
    def test_numbered_platter_phrase_matches_numbered_menu_item(self):
        repo = _build_repo()

        result = repo.resolve_menu_query("49 platter", limit=5)

        self.assertEqual(result.type, MenuQueryType.ITEM)
        self.assertIsNotNone(result.item)
        self.assertEqual(result.item.name, "49. Seafood Combo Platter")


if __name__ == "__main__":
    unittest.main()
