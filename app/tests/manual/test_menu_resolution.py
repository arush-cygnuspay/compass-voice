# tests/manual/test_menu_resolution.py

from pathlib import Path

from app.menu.store import MenuStore
from app.menu.repository import MenuRepository
from app.menu.exceptions import MenuLoadError
from app.menu.models import MenuItem


# =================================================
# CONFIG — PROJECT ROOT SAFE
# =================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "restaurants"

RESTAURANT_ID = "demo"

BASE_PATH = DATA_ROOT / RESTAURANT_ID
MENU_PATH = BASE_PATH / "menu.json"
ENTITY_INDEX_PATH = BASE_PATH / "entity_index.json"

TEST_QUERIES = [
    "chicken taco",
    "chick taco",
    "ice mocha",
    "iced mocha",
    "sprite",
    "chkn taco",
    "coke",
    "taco chicken",
    "random nonsense item",
]

# =================================================
# HELPERS
# =================================================

def print_item_full(item: MenuItem) -> None:
    """
    Pretty-print full MenuItem model.
    """
    print("\n  🔎 FULL ITEM OBJECT")
    print("  ------------------")

    print(f"  item_id   : {item.item_id}")
    print(f"  name      : {item.name}")
    print(f"  aliases   : {item.aliases}")
    print(f"  available : {item.available}")

    # -----------------
    # Pricing
    # -----------------
    print("\n  💰 PRICING")
    pricing = item.pricing
    print(f"    mode      : {pricing.mode}")
    print(f"    currency  : {pricing.currency}")

    if pricing.mode == "fixed":
        print(f"    price     : {pricing.price_cents}")

    elif pricing.mode == "variant":
        for v in pricing.variants or []:
            print(f"    - {v.label} ({v.variant_id}) → {v.price_cents}")

    elif pricing.mode == "unit":
        print(f"    unit price: {pricing.price_cents}")

    # -----------------
    # Side Groups
    # -----------------
    print("\n  🍟 SIDE GROUPS")
    if not item.side_groups:
        print("    (none)")
    else:
        for g in item.side_groups:
            print(f"    group_id     : {g.group_id}")
            print(f"    name         : {g.name}")
            print(f"    required     : {g.is_required}")
            print(f"    min/max      : {g.min_selector} / {g.max_selector}")
            print("    choices:")
            for c in g.choices:
                p = c.pricing
                price = (
                    p.price_cents
                    if p.mode == "fixed"
                    else "variant"
                )
                print(f"      - {c.name} ({c.item_id}) price={price}")
            print()

    # -----------------
    # Modifier Groups
    # -----------------
    print("\n  🧀 MODIFIER GROUPS")
    if not item.modifier_groups:
        print("    (none)")
    else:
        for g in item.modifier_groups:
            print(f"    group_id     : {g.group_id}")
            print(f"    name         : {g.name}")
            print(f"    required     : {g.is_required}")
            print(f"    min/max      : {g.min_selector} / {g.max_selector}")
            print("    choices:")
            for c in g.choices:
                print(
                    f"      - {c.name} ({c.modifier_id}) "
                    f"price_cents={c.price_cents}"
                )
            print()


# =================================================
# TEST RUNNER
# =================================================

def main():
    print("=== MENU RESOLUTION FULL INSPECTION TEST ===\n")

    try:
        store = MenuStore(MENU_PATH, ENTITY_INDEX_PATH)
        repo = MenuRepository(store)
    except MenuLoadError as e:
        print("❌ Failed to load menu:", e)
        return

    print("✅ Menu loaded successfully")
    print(f"   Items loaded      : {len(store.items)}")
    print(f"   Categories loaded : {len(store.categories)}\n")

    for query in TEST_QUERIES:
        print("=" * 60)
        print(f"Query: '{query}'\n")

        # ---- Tier 1: Entity index ----
        entity_matches = store.find_entity(query, allowed_types={"item"})
        if entity_matches:
            ent = entity_matches[0]
            print("  ✔ ENTITY INDEX MATCH")
            print(f"    entity → {ent}")

            item = store.get_item(ent["item_id"])
            print_item_full(item)
            continue

        # ---- Tier 2: Token overlap ----
        token_item = store.find_item_by_tokens(query)
        if token_item:
            print("  ✔ TOKEN OVERLAP MATCH")
            print_item_full(token_item)
            continue

        print("  ❌ NO MATCH FOUND")

    print("\n=== TEST COMPLETE ===")


if __name__ == "__main__":
    main()
