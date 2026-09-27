"""One-off migration of the active demo datasets to the four-item canonical menu.

Run once with:  python scripts/migrate_menu_data.py
Rewrites the active item-level/daily datasets so they contain only the four
canonical menu items and seeds a raw ingredient inventory. Historical
redistribution records are never touched.
"""

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.kitchen_config import (  # noqa: E402
    INGREDIENT_UNITS,
    MENU_ITEMS,
    PROTOTYPE_NOTE,
    RECIPES,
)
from src.raw_inventory import (  # noqa: E402
    RawInventoryError,
    add_raw_batch,
    record_unmapped_batch,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
START = date(2026, 6, 28)
END = date(2026, 9, 26)

# Plausible prototype demand shape for a four-item menu kitchen.
ITEM_BASE = {
    "Chai": 430,
    "Rice with Chicken": 245,
    "Rice with Tomato": 165,
    "Rice with Dal": 150,
}
ITEM_TREND = {
    "Chai": 0.55,
    "Rice with Chicken": 0.20,
    "Rice with Tomato": -0.10,
    "Rice with Dal": -0.15,
}
WEATHERS = ["Sunny", "Cloudy", "Rainy"]
PRECIPITATION = {
    "Sunny": ["No precipitation"],
    "Cloudy": ["No Rain", "Light Rain"],
    "Rainy": ["Light Rain", "Heavy Rain"],
}

rng = np.random.default_rng(20260926)

# Prototype raw stock, anchored to the demo scenario date (2026-09-27).
# Deliberately awkward on purpose: extra whitespace, mixed case, two Rice
# batches for FEFO, one expiring-soon batch, one already-expired batch, and one
# unknown ingredient that must be preserved and flagged rather than deleted.
RAW_SEED = [
    ("  tea leaves ", 0.5, "kg", date(2026, 9, 25), date(2027, 1, 18), True),
    ("Sugar", 25.0, "kg", date(2026, 9, 25), date(2027, 3, 19), True),
    ("  milk ", 40000.0, "ml", date(2026, 9, 26), date(2026, 10, 5), True),
    ("Rice", 60.0, "kg", date(2026, 9, 1), date(2026, 12, 15), True),
    ("Rice", 45.0, "kg", date(2026, 9, 10), date(2027, 1, 5), True),
    ("Dal", 22.0, "kg", date(2026, 9, 20), date(2027, 4, 8), True),
    ("Tomato", 9.0, "kg", date(2026, 9, 25), date(2026, 9, 30), True),
    ("chicken", 14.0, "kg", date(2026, 9, 20), date(2026, 9, 25), True),
    ("Chocolate", 3.0, "kg", date(2026, 9, 20), date(2026, 12, 19), False),
]



def build_rows():
    days = (END - START).days + 1
    daily_rows, item_rows = [], []
    for offset in range(days):
        day = START + timedelta(days=offset)
        dow = day.weekday()
        holiday = 1 if dow in (5, 6) else 0
        special = 1 if (day.day % 17 == 0) else 0
        weather = WEATHERS[(day.day + offset) % len(WEATHERS)]
        precipitation = PRECIPITATION[weather][day.day % len(PRECIPITATION[weather])]

        predicted_total = produced_total = consumed_total = 0
        for item in MENU_ITEMS:
            level = (
                ITEM_BASE[item]
                + ITEM_TREND[item] * offset
                + (60 if dow >= 5 else 0)
                + (40 if special else 0)
                + rng.normal(0, 22)
            )
            level = max(level, 40.0)
            predicted = int(round(level))
            produced = int(round(predicted * rng.uniform(1.02, 1.11)))
            consumed = min(int(round(predicted * rng.uniform(0.97, 1.01))), produced)
            predicted_total += predicted
            produced_total += produced
            consumed_total += consumed
            item_rows.append(
                {
                    "date": day.isoformat(),
                    "menu_item": item,
                    "predicted_quantity": predicted,
                    "actual_production": produced,
                    "actual_consumption": consumed,
                    "surplus_quantity": produced - consumed,
                }
            )
        daily_rows.append(
            {
                "date": day.isoformat(),
                "predicted_demand": predicted_total,
                "actual_production": produced_total,
                "actual_consumption": consumed_total,
                "surplus_quantity": produced_total - consumed_total,
                "holiday": holiday,
                "special_event": special,
                "weather": weather,
                "precipitation": precipitation,
            }
        )
    return daily_rows, item_rows


def write_operational_files(daily_rows, item_rows):
    pd.DataFrame(daily_rows).to_csv(DATA_DIR / "daily_history.csv", index=False)
    pd.DataFrame(item_rows).to_csv(DATA_DIR / "item_history.csv", index=False)
    pd.DataFrame(
        [
            {
                "date": "2026-09-27",
                "planned_quantity": sum(ITEM_BASE.values()) + 30,
                "holiday": 0,
                "special_event": 0,
                "weather": "Sunny",
                "precipitation": "No precipitation",
                "safety_buffer": 5.0,
            }
        ]
    ).to_csv(DATA_DIR / "daily_operations.csv", index=False)
    # Surplus shelf life and surplus batches are per canonical menu item now.
    pd.DataFrame(
        [
            {"menu_item": "Chai", "shelf_life_days": 1},
            {"menu_item": "Rice with Chicken", "shelf_life_days": 1},
            {"menu_item": "Rice with Tomato", "shelf_life_days": 1},
            {"menu_item": "Rice with Dal", "shelf_life_days": 2},
        ]
    ).to_csv(DATA_DIR / "surplus_shelf_life.csv", index=False)
    added = date(2026, 9, 26)
    surplus_rows = []
    for index, (item, qty) in enumerate(
        [("Chai", 14), ("Rice with Chicken", 9), ("Rice with Tomato", 6), ("Rice with Dal", 4)],
        start=1,
    ):
        shelf = 1 if item != "Rice with Dal" else 2
        surplus_rows.append(
            {
                "inventory_id": f"INV-{added.strftime('%Y%m%d')}-{index:03d}",
                "date_added": added.isoformat(),
                "menu_item": item,
                "original_quantity": qty,
                "remaining_quantity": qty,
                "expiry_date": (added + timedelta(days=shelf)).isoformat(),
                "status": "Available",
            }
        )
    pd.DataFrame(surplus_rows).to_csv(DATA_DIR / "inventory.csv", index=False)


def seed_raw_inventory():
    raw_path = DATA_DIR / "raw_inventory.csv"
    if raw_path.exists():
        raw_path.unlink()
    for ingredient, quantity, unit, added, expires, mapped in RAW_SEED:
        try:
            if mapped:
                add_raw_batch(raw_path, ingredient, quantity, unit, added, expires)
            else:
                record_unmapped_batch(raw_path, ingredient, quantity, unit, added, expires)
        except RawInventoryError as error:
            print(f"seed '{ingredient}' -> {error}")
    usage_path = DATA_DIR / "raw_usage.csv"
    if usage_path.exists():
        usage_path.unlink()


def main():
    daily_rows, item_rows = build_rows()
    write_operational_files(daily_rows, item_rows)
    seed_raw_inventory()
    print("MENU:", MENU_ITEMS)
    print("UNITS:", INGREDIENT_UNITS)
    print("RECIPES:", RECIPES)
    print("NOTE:", PROTOTYPE_NOTE)
    print("daily rows:", len(daily_rows), "item rows:", len(item_rows))


if __name__ == "__main__":
    main()
