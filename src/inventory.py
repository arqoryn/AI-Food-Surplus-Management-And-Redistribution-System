"""Operational surplus inventory (Phase E).

Holds batches of surplus food created automatically from End-of-Day results
that have not yet been redistributed. This subsystem is purely operational:
it is never used by the demand model or any ML training data.
"""

from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Mapping

import pandas as pd

from src.data_preprocessing import (
    INVENTORY_COLUMNS,
    SHELF_LIFE_COLUMNS,
    load_csv,
)

DEFAULT_SHELF_LIFE_DAYS = 1
EXPIRING_SOON_DAYS = 2

STATUS_AVAILABLE = "Available"
STATUS_EXPIRING_SOON = "Expiring Soon"
STATUS_EXPIRED = "Expired"
STATUS_DEPLETED = "Depleted"


def load_inventory(path: Path) -> pd.DataFrame:
    """Load the inventory dataset; returns an empty frame when no file exists yet."""
    if not Path(path).exists():
        return pd.DataFrame(columns=INVENTORY_COLUMNS)
    return load_csv(Path(path), INVENTORY_COLUMNS)


def load_shelf_life(path: Path) -> Dict[str, int]:
    """Load the demo shelf-life configuration as {menu_item: shelf_life_days}."""
    if not Path(path).exists():
        return {}
    data = load_csv(Path(path), SHELF_LIFE_COLUMNS)
    shelf_life = {}
    for _, row in data.iterrows():
        try:
            days = int(row["shelf_life_days"])
        except (TypeError, ValueError):
            continue
        if days > 0 and str(row["menu_item"]).strip():
            shelf_life[str(row["menu_item"]).strip()] = days
    return shelf_life


def calculate_expiry(date_added: date, shelf_life_days: int) -> date:
    """Expiry = date_added + configured shelf_life_days."""
    days = int(shelf_life_days) if int(shelf_life_days) > 0 else DEFAULT_SHELF_LIFE_DAYS
    return date_added + timedelta(days=days)


def calculate_status(remaining_quantity, expiry_date, today: date) -> str:
    """Status derived from remaining quantity, expiry date, and today's date."""
    if int(remaining_quantity) <= 0:
        return STATUS_DEPLETED
    expiry = pd.Timestamp(expiry_date).date()
    if today > expiry:
        return STATUS_EXPIRED
    if (expiry - today).days <= EXPIRING_SOON_DAYS:
        return STATUS_EXPIRING_SOON
    return STATUS_AVAILABLE


def create_inventory_batches(
    date_added: date,
    surplus_by_item: Mapping[str, int],
    shelf_life_days: Mapping[str, int],
    existing_inventory: pd.DataFrame = None,
) -> pd.DataFrame:
    """Build one inventory batch per positive item surplus for a completed day."""
    existing = existing_inventory if existing_inventory is not None else pd.DataFrame(columns=INVENTORY_COLUMNS)
    existing_ids = set(existing["inventory_id"].astype(str))
    date_prefix = f"INV-{date_added.strftime('%Y%m%d')}-"
    sequence = sum(1 for inventory_id in existing_ids if str(inventory_id).startswith(date_prefix))

    rows = []
    for menu_item in sorted(surplus_by_item):
        surplus = int(surplus_by_item[menu_item])
        if surplus <= 0:
            continue
        configured_days = int(shelf_life_days.get(menu_item, DEFAULT_SHELF_LIFE_DAYS))
        expiry = calculate_expiry(date_added, configured_days)
        inventory_id = f"{date_prefix}{sequence + 1:03d}"
        sequence += 1
        while inventory_id in existing_ids:
            sequence += 1
            inventory_id = f"{date_prefix}{sequence:03d}"
        existing_ids.add(inventory_id)
        rows.append(
            {
                "inventory_id": inventory_id,
                "date_added": date_added.isoformat(),
                "menu_item": menu_item,
                "original_quantity": surplus,
                "remaining_quantity": surplus,
                "expiry_date": expiry.isoformat(),
                "status": calculate_status(surplus, expiry, date_added),
            }
        )
    if not rows:
        return pd.DataFrame(columns=INVENTORY_COLUMNS)
    return pd.DataFrame(rows, columns=INVENTORY_COLUMNS)


def decrement_remaining(remaining_quantity, quantity) -> int:
    """Future Phase F helper: reduce a batch's remaining quantity safely.

    Not wired into Recipient Matching yet (Phase F will do the decrementing).
    """
    remaining = int(remaining_quantity)
    amount = int(quantity)
    if amount <= 0:
        raise ValueError("quantity must be positive.")
    if remaining <= 0:
        raise ValueError("this inventory batch is already depleted.")
    if amount > remaining:
        raise ValueError("quantity exceeds the remaining inventory for this batch.")
    return remaining - amount


def apply_inventory_decrement(path: Path, inventory_id: str, quantity) -> int:
    """Persist a confirmed redistribution against one inventory batch (Phase F).

    Decreases remaining_quantity (never below zero), keeps original_quantity,
    date_added and expiry_date untouched, recalculates the status with the
    existing Phase E rules, and updates the batch in place (never deletes it).
    Returns the new remaining quantity.
    """
    inventory = load_inventory(path)
    matches = inventory[inventory["inventory_id"].astype(str) == str(inventory_id)]
    if matches.empty:
        raise ValueError(f"Inventory batch {inventory_id} was not found.")
    row_index = matches.index[0]
    new_remaining = decrement_remaining(int(inventory.at[row_index, "remaining_quantity"]), quantity)
    inventory.at[row_index, "remaining_quantity"] = new_remaining
    inventory.at[row_index, "status"] = calculate_status(
        new_remaining, inventory.at[row_index, "expiry_date"], pd.Timestamp.today().date()
    )
    inventory.to_csv(path, index=False)
    return new_remaining



def update_active_surplus_shelf_life(
    path: Path,
    shelf_life_days: Mapping[str, int],
    today: date = None,
) -> int:
    """Update expiry dates for active/non-depleted surplus inventory batches.

    Only operational/active batches (remaining_quantity > 0 and status != Depleted)
    have their expiry_date recomputed as: date_added + configured shelf_life_days.
    Status is also recalculated. Depleted batches and historical records remain untouched.
    Returns the number of batches updated.
    """
    path = Path(path)
    if not path.exists():
        return 0
    inventory = load_inventory(path)
    if inventory.empty:
        return 0

    eval_today = today if today is not None else pd.Timestamp.today().date()
    updated_count = 0

    for idx, row in inventory.iterrows():
        try:
            remaining = int(row.get("remaining_quantity", 0))
        except (TypeError, ValueError):
            remaining = 0
        status = str(row.get("status", "")).strip()

        # Only update active/non-depleted batches
        if remaining <= 0 or status == STATUS_DEPLETED:
            continue

        item = str(row.get("menu_item", "")).strip()
        if item in shelf_life_days:
            days = int(shelf_life_days[item])
            try:
                added = pd.Timestamp(row["date_added"]).date()
            except (TypeError, ValueError):
                continue
            new_expiry = calculate_expiry(added, days)
            inventory.at[idx, "expiry_date"] = new_expiry.isoformat()
            inventory.at[idx, "status"] = calculate_status(remaining, new_expiry, eval_today)
            updated_count += 1

    if updated_count > 0:
        inventory.to_csv(path, index=False)
    return updated_count
