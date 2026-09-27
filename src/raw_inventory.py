"""Raw ingredient inventory: batches, normalisation, FEFO, capacity, usage.

This subsystem is strictly upstream of food production. It is operational only
and is never used by the demand model.

Every public entry point degrades gracefully: expected data problems raise a
narrow, typed ``RawInventoryError`` that the dashboard renders as a warning,
rather than an unhandled exception that would take the whole app down.
"""

import re
from datetime import date
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import pandas as pd

from src.kitchen_config import (
    APPROACHING_EXPIRY_DAYS,
    EXPIRING_SOON_DAYS,
    INGREDIENT_UNITS,
    MENU_ITEMS,
    RAW_INGREDIENTS,
    TONE_APPROACHING,
    TONE_DEPLETED,
    TONE_EXPIRED,
    TONE_EXPIRING,
    TONE_HEALTHY,
    TONE_UNKNOWN,
    UNIT_TO_CANONICAL,
    ingredient_unit,
    menu_items_using,
    recipe_for,
    unit_supported,
)

RAW_INVENTORY_COLUMNS = [
    "raw_id", "ingredient", "canonical_ingredient", "unit",
    "original_quantity", "remaining_quantity",
    "date_added", "expiry_date", "status", "mapping_note",
]

RAW_USAGE_COLUMNS = [
    "usage_id", "date", "menu_item", "ingredient",
    "quantity_used", "unit", "production_reference", "raw_id",
]

STATUS_AVAILABLE = "Available"
STATUS_EXPIRING_SOON = "Expiring Soon"
STATUS_EXPIRED = "Expired"
STATUS_DEPLETED = "Depleted"

MAPPED_OK = "Mapped"
MAPPED_UNKNOWN = "Unknown / Unmapped"

# Deterministic normalisation index: collapsed-lowercase -> canonical name.
# Only case and whitespace differences resolve here; arbitrary misspellings
# (e.g. "Milkk") stay Unknown on purpose, because inventory data must never be
# silently rewritten into a guess.
_CANONICAL_INDEX: Dict[str, str] = {
    re.sub(r"\s+", " ", name.strip().lower()): name for name in RAW_INGREDIENTS
}

# Only these are accepted as explicit admin corrections; everything else stays
# Unknown rather than being guessed.
EXPLICIT_ALIASES: Dict[str, str] = {
    "tea": "Tea Leaves",
    "tea leaf": "Tea Leaves",
}


class RawInventoryError(Exception):
    """Expected, user-facing raw-inventory problem."""



# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalize_ingredient(value) -> Tuple[str, str]:
    """Return (canonical_ingredient_or_blank, mapping_note).

    Handles leading/trailing/repeated whitespace and case. Never guesses
    arbitrary misspellings.
    """
    text = "" if value is None else str(value)
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return "", "Missing ingredient name"
    key = cleaned.lower()
    if key in _CANONICAL_INDEX:
        return _CANONICAL_INDEX[key], MAPPED_OK
    if key in EXPLICIT_ALIASES:
        return EXPLICIT_ALIASES[key], MAPPED_OK
    return "", f"'{cleaned}' is not one of the seven canonical raw ingredients"


def is_mapped(value) -> bool:
    return normalize_ingredient(value)[0] != ""


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def parse_positive_quantity(value, label: str = "Quantity") -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise RawInventoryError(f"{label} must be a number, got '{value}'.")
    if number != number or number in (float("inf"), float("-inf")):
        raise RawInventoryError(f"{label} must be a finite number.")
    if number <= 0:
        raise RawInventoryError(f"{label} must be greater than zero.")
    return number


def parse_date(value, label: str):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        raise RawInventoryError(f"{label} is required.")
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError):
        raise RawInventoryError(f"{label} is not a valid date: '{value}'.")
    if pd.isna(parsed):
        raise RawInventoryError(f"{label} is not a valid date: '{value}'.")
    return parsed.date()


def validate_unit(unit: str, canonical_ingredient: str) -> Tuple[str, float]:
    """Validate a unit and return (canonical_unit, conversion_factor)."""
    cleaned = str(unit or "").strip()
    if not cleaned:
        raise RawInventoryError("Unit is required.")
    if not unit_supported(cleaned):
        supported = ", ".join(sorted(unit for unit, _ in UNIT_TO_CANONICAL))
        raise RawInventoryError(f"Unsupported unit '{cleaned}'. Supported units: {supported}.")
    target = ingredient_unit(canonical_ingredient)
    factor = UNIT_TO_CANONICAL.get((cleaned, target))
    if factor is None:
        raise RawInventoryError(
            f"Unit '{cleaned}' cannot be converted for {canonical_ingredient} "
            f"(canonical unit: {target})."
        )
    return target, factor


def raw_status(remaining, expiry_date, today: date) -> str:
    if float(remaining) <= 0:
        return STATUS_DEPLETED
    if today > pd.Timestamp(expiry_date).date():
        return STATUS_EXPIRED
    if (pd.Timestamp(expiry_date).date() - today).days <= EXPIRING_SOON_DAYS:
        return STATUS_EXPIRING_SOON
    return STATUS_AVAILABLE




# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def load_raw_inventory(path: Path) -> pd.DataFrame:
    """Load raw batches. Returns an empty frame for a missing/empty/corrupt file."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=RAW_INVENTORY_COLUMNS)
    try:
        data = pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError):
        return pd.DataFrame(columns=RAW_INVENTORY_COLUMNS)
    if data.empty:
        return pd.DataFrame(columns=RAW_INVENTORY_COLUMNS)
    data.columns = [str(c).replace("\ufeff", "").strip() for c in data.columns]
    for column in RAW_INVENTORY_COLUMNS:
        if column not in data.columns:
            data[column] = pd.NA
    return data[RAW_INVENTORY_COLUMNS]


def save_raw_inventory(path: Path, data: pd.DataFrame) -> None:
    if data is None or data.empty:
        pd.DataFrame(columns=RAW_INVENTORY_COLUMNS).to_csv(path, index=False)
        return
    data.to_csv(path, index=False)


def next_raw_id(existing: pd.DataFrame, today: date) -> str:
    prefix = f"RAW-{today.strftime('%Y%m%d')}-"
    used = set(existing["raw_id"].astype(str)) if not existing.empty else set()
    sequence = sum(1 for value in used if value.startswith(prefix)) + 1
    while f"{prefix}{sequence:03d}" in used:
        sequence += 1
    return f"{prefix}{sequence:03d}"


def add_raw_batch(
    path: Path,
    ingredient: str,
    quantity,
    unit: str,
    date_added,
    expiry_date,
    note: str = "",
) -> dict:
    """Validate and persist one mapped raw ingredient batch."""
    canonical, mapping_note = normalize_ingredient(ingredient)
    if not canonical:
        raise RawInventoryError(mapping_note)
    amount = parse_positive_quantity(quantity)
    target_unit, factor = validate_unit(unit, canonical)
    added = parse_date(date_added, "Date added")
    expires = parse_date(expiry_date, "Expiry date")
    if expires < added:
        raise RawInventoryError("Expiry date cannot be earlier than the date added.")

    canonical_amount = round(amount * factor, 6)
    existing = load_raw_inventory(path)
    record = {
        "raw_id": next_raw_id(existing, added),
        "ingredient": canonical,
        "canonical_ingredient": canonical,
        "unit": target_unit,
        "original_quantity": canonical_amount,
        "remaining_quantity": canonical_amount,
        "date_added": added.isoformat(),
        "expiry_date": expires.isoformat(),
        "status": raw_status(canonical_amount, expires, added),
        "mapping_note": note or MAPPED_OK,
    }
    combined = (
        pd.concat([existing, pd.DataFrame([record])], ignore_index=True)
        if not existing.empty
        else pd.DataFrame([record])
    )
    save_raw_inventory(path, combined)
    return record


def record_unmapped_batch(
    path: Path,
    ingredient: str,
    quantity,
    unit: str,
    date_added,
    expiry_date,
) -> dict:
    """Persist a user-entered ingredient verbatim, flagged as unmapped.

    The record is preserved exactly as typed. It is excluded from every recipe
    and capacity calculation but stays visible and removable by the admin.
    """
    text = re.sub(r"\s+", " ", str(ingredient or "")).strip()
    if not text:
        raise RawInventoryError("Ingredient name is required.")
    amount = parse_positive_quantity(quantity)
    cleaned_unit = str(unit or "kg").strip()
    if not unit_supported(cleaned_unit):
        supported = ", ".join(sorted(unit for unit, _ in UNIT_TO_CANONICAL))
        raise RawInventoryError(f"Unsupported unit '{cleaned_unit}'. Supported units: {supported}.")
    added = parse_date(date_added, "Date added")
    expires = parse_date(expiry_date, "Expiry date")
    if expires < added:
        raise RawInventoryError("Expiry date cannot be earlier than the date added.")
    existing = load_raw_inventory(path)
    record = {
        "raw_id": next_raw_id(existing, added),
        "ingredient": text,
        "canonical_ingredient": "",
        "unit": cleaned_unit,
        "original_quantity": amount,
        "remaining_quantity": amount,
        "date_added": added.isoformat(),
        "expiry_date": expires.isoformat(),
        "status": raw_status(amount, expires, added),
        "mapping_note": MAPPED_UNKNOWN,
    }
    combined = (
        pd.concat([existing, pd.DataFrame([record])], ignore_index=True)
        if not existing.empty
        else pd.DataFrame([record])
    )
    save_raw_inventory(path, combined)
    return record


def update_raw_batch(
    path: Path,
    raw_id: str,
    ingredient: str,
    quantity,
    unit: str,
    date_added,
    expiry_date,
) -> dict:
    """Edit an existing batch in place, applying the same validation as Add.

    The internal raw_id is preserved so FEFO, usage history and idempotency
    stay stable. Fixing a misspelled ingredient re-maps it immediately, so the
    batch becomes eligible for recipes and capacity without creating a new row.
    """
    existing = load_raw_inventory(path)
    if existing.empty:
        raise RawInventoryError("Raw inventory is empty.")
    matches = existing[existing["raw_id"].astype(str) == str(raw_id)]
    if matches.empty:
        raise RawInventoryError(f"Raw batch {raw_id} was not found.")
    row_index = matches.index[0]

    canonical, mapping_note = normalize_ingredient(ingredient)
    text = re.sub(r"\s+", " ", str(ingredient or "")).strip()
    if not text:
        raise RawInventoryError("Ingredient name is required.")
    amount = parse_positive_quantity(quantity)
    added = parse_date(date_added, "Date added")
    expires = parse_date(expiry_date, "Expiry date")
    if expires < added:
        raise RawInventoryError("Expiry date cannot be earlier than the date added.")

    if canonical:
        target_unit, factor = validate_unit(unit, canonical)
        stored_quantity = round(amount * factor, 6)
        note = MAPPED_OK
    else:
        # Unknown ingredients are preserved exactly as re-typed, never guessed.
        cleaned_unit = str(unit or "kg").strip()
        if not unit_supported(cleaned_unit):
            supported = ", ".join(sorted(u for u, _ in UNIT_TO_CANONICAL))
            raise RawInventoryError(
                f"Unsupported unit '{cleaned_unit}'. Supported units: {supported}."
            )
        target_unit = cleaned_unit
        stored_quantity = amount
        note = MAPPED_UNKNOWN

    existing.at[row_index, "ingredient"] = text
    existing.at[row_index, "canonical_ingredient"] = canonical
    existing.at[row_index, "unit"] = target_unit
    existing.at[row_index, "original_quantity"] = stored_quantity
    # Editing restates the batch, so remaining tracks the corrected quantity.
    existing.at[row_index, "remaining_quantity"] = stored_quantity
    existing.at[row_index, "date_added"] = added.isoformat()
    existing.at[row_index, "expiry_date"] = expires.isoformat()
    existing.at[row_index, "status"] = raw_status(stored_quantity, expires, added)
    existing.at[row_index, "mapping_note"] = note
    save_raw_inventory(path, existing)
    return existing.loc[row_index].to_dict()


def remove_raw_batch(path: Path, raw_id: str) -> None:
    """Admin-only removal of any batch. Nothing is ever removed automatically.

    Historical usage rows that reference this batch are deliberately left
    untouched, so past production records stay intact.
    """
    existing = load_raw_inventory(path)
    if existing.empty:
        raise RawInventoryError("Raw inventory is empty.")
    matches = existing[existing["raw_id"].astype(str) == str(raw_id)]
    if matches.empty:
        raise RawInventoryError(f"Raw batch {raw_id} was not found.")
    save_raw_inventory(path, existing.drop(index=matches.index).reset_index(drop=True))


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

def usable_batches(data: pd.DataFrame, today: date) -> pd.DataFrame:
    """Batches usable for production (not expired, not depleted), FEFO ordered."""
    if data is None or data.empty:
        return pd.DataFrame(columns=RAW_INVENTORY_COLUMNS)
    frame = data.copy()
    frame["remaining_quantity"] = pd.to_numeric(frame["remaining_quantity"], errors="coerce").fillna(0)
    frame["expiry_date"] = pd.to_datetime(frame["expiry_date"], errors="coerce")
    usable = frame[
        (frame["remaining_quantity"] > 0)
        & frame["expiry_date"].notna()
        & (frame["expiry_date"].dt.date >= today)
    ]
    return usable.sort_values(["expiry_date", "raw_id"], kind="stable")


def status_tone(days_remaining, remaining, status: str, expiring_days=None, approaching_days=None) -> str:
    """Map a batch to a display tone. Thresholds are centralized, not ad hoc.

    ``days_remaining`` is already clamped at 0, so an expired batch reads as
    "0 days left" rather than a confusing negative number. Thresholds default to
    the centralized config and are overridable from Settings.
    """
    expiring_limit = EXPIRING_SOON_DAYS if expiring_days is None else int(expiring_days)
    approaching_limit = APPROACHING_EXPIRY_DAYS if approaching_days is None else int(approaching_days)
    if status == STATUS_DEPLETED or (remaining is not None and float(remaining) <= 0):
        return TONE_DEPLETED
    if days_remaining is None:
        return TONE_UNKNOWN
    if status == STATUS_EXPIRED or days_remaining < 0:
        return TONE_EXPIRED
    if days_remaining <= expiring_limit:
        return TONE_EXPIRING
    if days_remaining <= approaching_limit:
        return TONE_APPROACHING
    return TONE_HEALTHY


def batch_label(date_added, sequence) -> str:
    """Human-readable batch name, e.g. '20 Sep 2026 - Batch 02'.

    The internal RAW-YYYYMMDD-XXX id is never changed; this is display only.
    """
    try:
        stamp = pd.Timestamp(date_added).strftime("%d %b %Y")
    except (TypeError, ValueError):
        stamp = "Unknown date"
    return f"{stamp} - Batch {int(sequence):02d}"


def batch_sequence(raw_id: str) -> int:
    """Trailing sequence number from an internal raw id, for display only."""
    parts = str(raw_id).split("-")
    try:
        return int(parts[-1])
    except (TypeError, ValueError):
        return 0


def stock_table(data: pd.DataFrame, today: date, expiring_days=None, approaching_days=None) -> pd.DataFrame:
    """Display view with recomputed status, clamped days-to-expiry and tone.

    Threshold overrides come from Settings; omitted means use the centralized
    configuration defaults.
    """
    if data is None or data.empty:
        return pd.DataFrame(columns=RAW_INVENTORY_COLUMNS)
    frame = data.copy()
    frame["remaining_quantity"] = pd.to_numeric(frame["remaining_quantity"], errors="coerce")
    frame["original_quantity"] = pd.to_numeric(frame["original_quantity"], errors="coerce")
    frame["expiry_date"] = pd.to_datetime(frame["expiry_date"], errors="coerce")
    frame["date_added"] = pd.to_datetime(frame["date_added"], errors="coerce")
    statuses, days, tones = [], [], []
    for _, row in frame.iterrows():
        if pd.isna(row["expiry_date"]):
            statuses.append("Unknown")
            days.append(None)
            tones.append(TONE_UNKNOWN)
            continue
        expiry = row["expiry_date"].date()
        status = raw_status(row["remaining_quantity"], expiry, today)
        # Never surface a negative count: expired and expiring-today both read 0.
        remaining_days = max((expiry - today).days, 0)
        statuses.append(status)
        days.append(remaining_days)
        tones.append(
            status_tone(remaining_days, row["remaining_quantity"], status, expiring_days, approaching_days)
        )
    frame["status"] = statuses
    frame["days_to_expiry"] = days
    frame["status_tone"] = tones
    frame["batch_label"] = [
        batch_label(row["date_added"], batch_sequence(row["raw_id"]))
        for _, row in frame.iterrows()
    ]
    return frame.sort_values(
        ["expiry_date", "raw_id"], kind="stable", na_position="last"
    ).reset_index(drop=True)


def available_quantity(data: pd.DataFrame, ingredient: str, today: date) -> float:
    """Usable quantity of one canonical ingredient."""
    usable = usable_batches(data, today)
    if usable.empty:
        return 0.0
    rows = usable[usable["canonical_ingredient"].astype(str) == str(ingredient)]
    return float(rows["remaining_quantity"].sum())


def expiring_soon(data: pd.DataFrame, today: date, expiring_days=None, approaching_days=None) -> List[dict]:
    """Batches needing expiry attention, across all three urgency tiers.

    Tiers come from the centralized thresholds (or Settings overrides): expired,
    expiring soon, and approaching expiry. Healthy batches are deliberately
    excluded so the Notification Center stays meaningful.
    """
    view = stock_table(data, today, expiring_days, approaching_days)
    if view.empty:
        return []
    flagged = view[
        view["status_tone"].isin([TONE_EXPIRED, TONE_EXPIRING, TONE_APPROACHING])
        & (view["remaining_quantity"] > 0)
    ]
    notices = []
    for _, row in flagged.iterrows():
        ingredient = str(row["canonical_ingredient"])
        days_left = row["days_to_expiry"]
        expired_days = 0
        if str(row["status"]) == STATUS_EXPIRED:
            try:
                expired_days = max((today - row["expiry_date"].date()).days, 0)
            except (TypeError, ValueError):
                expired_days = 0
        notices.append(
            {
                "ingredient": ingredient if ingredient else str(row["ingredient"]),
                "status": str(row["status"]),
                "tone": str(row["status_tone"]),
                "days_left": days_left,
                "expired_days_ago": expired_days,
                "remaining": float(row["remaining_quantity"]),
                "unit": str(row["unit"]),
                "expiry_date": row["expiry_date"],
                "menu_options": menu_items_using(ingredient) if ingredient else [],
            }
        )
    return notices


# ---------------------------------------------------------------------------
# Capacity
# ---------------------------------------------------------------------------

def serving_capacity(data: pd.DataFrame, menu_item: str, today: date) -> Dict[str, object]:
    """Servings of one menu item supported right now (bottleneck ingredient)."""
    result: Dict[str, object] = {
        "menu_item": menu_item,
        "capacity": 0,
        "bottleneck": None,
        "recipe_missing": False,
        "recipe_incomplete": False,
        "issue": "",
        "ingredient_support": {},
    }
    if str(menu_item) not in MENU_ITEMS:
        result["recipe_missing"] = True
        result["issue"] = (
            f"'{menu_item}' is not a canonical menu item, so capacity cannot be calculated."
        )
        return result
    recipe = recipe_for(menu_item)
    if not recipe:
        result["recipe_missing"] = True
        result["issue"] = f"No recipe is defined for {menu_item}, so capacity cannot be calculated."
        return result
    incomplete = [name for name, amount in recipe if not amount or float(amount) <= 0]
    if incomplete:
        result["recipe_incomplete"] = True
        result["issue"] = (
            f"Recipe for {menu_item} is incomplete (no quantity for: {', '.join(incomplete)}), "
            "so capacity cannot be reliably calculated."
        )
        return result

    support: Dict[str, dict] = {}
    bottleneck: Tuple[str, int] | None = None
    for name, per_serving in recipe:
        stock = available_quantity(data, name, today)
        servings = int(stock // per_serving) if per_serving else 0
        support[name] = {
            "available": round(stock, 4),
            "per_serving": per_serving,
            "unit": INGREDIENT_UNITS.get(name, "kg"),
            "servings": servings,
        }
        if bottleneck is None or servings < bottleneck[1]:
            bottleneck = (name, servings)
    result["ingredient_support"] = support
    result["capacity"] = max(bottleneck[1], 0) if bottleneck else 0
    result["bottleneck"] = bottleneck[0] if bottleneck else None
    starved = [name for name, info in support.items() if info["servings"] <= 0]
    if starved:
        result["issue"] = (
            f"No usable stock for {', '.join(starved)}. Capacity is 0 servings until "
            "these ingredients are replenished."
        )
    return result


def capacity_summary(data: pd.DataFrame, today: date) -> List[Dict[str, object]]:
    """Capacity for every canonical menu item."""
    return [serving_capacity(data, item, today) for item in MENU_ITEMS]


# ---------------------------------------------------------------------------
# FEFO
# ---------------------------------------------------------------------------

def plan_fefo_consumption(
    data: pd.DataFrame, requirements: Mapping[str, float], today: date
) -> Tuple[Dict[str, List[dict]], List[str]]:
    """Assign ingredient requirements to batches, earliest expiry first.

    Returns (plan keyed by ingredient, ingredients that could not be fully
    covered). Expired and depleted batches are never used.
    """
    plan: Dict[str, List[dict]] = {}
    shortfalls: List[str] = []
    usable = usable_batches(data, today)
    for ingredient, amount in requirements.items():
        needed = float(amount)
        picks: List[dict] = []
        if needed > 0 and not usable.empty:
            rows = usable[usable["canonical_ingredient"].astype(str) == str(ingredient)]


# ---------------------------------------------------------------------------
# Usage history
# ---------------------------------------------------------------------------

def load_raw_usage(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=RAW_USAGE_COLUMNS)
    try:
        data = pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError):
        return pd.DataFrame(columns=RAW_USAGE_COLUMNS)
    if data.empty:
        return pd.DataFrame(columns=RAW_USAGE_COLUMNS)
    data.columns = [str(c).replace("\ufeff", "").strip() for c in data.columns]
    for column in RAW_USAGE_COLUMNS:
        if column not in data.columns:
            data[column] = pd.NA
    return data[RAW_USAGE_COLUMNS]


def usage_reference(event_date) -> str:
    """Stable idempotency key for one production event."""
    return f"EOD-{pd.Timestamp(event_date).date().isoformat()}"


def usage_already_recorded(usage: pd.DataFrame, reference: str) -> bool:
    if usage is None or usage.empty:
        return False
    return usage["production_reference"].astype(str).eq(str(reference)).any()


def append_raw_usage(path: Path, rows: List[dict]) -> int:
    """Append usage rows, continuing the existing usage_id sequence."""
    if not rows:
        return 0
    existing = load_raw_usage(path)
    next_number = len(existing) + 1
    new = pd.DataFrame(rows)
    new["usage_id"] = [
        f"USG-{index:04d}" for index in range(next_number, next_number + len(new))
    ]
    new = new[RAW_USAGE_COLUMNS]
    combined = pd.concat([existing, new], ignore_index=True) if not existing.empty else new
    combined.to_csv(path, index=False)
    return len(new)


# ---------------------------------------------------------------------------
# Production deduction (FEFO, idempotent)
# ---------------------------------------------------------------------------

def deduct_for_production(
    raw_path: Path,
    usage_path: Path,
    event_date,
    production_by_item: Mapping[str, int],
) -> Dict[str, object]:
    """Deduct raw ingredients for CONFIRMED actual production.

    Each menu item's recipe requirement is satisfied FEFO, and the picks are
    attributed to that menu item so usage history stays per-item. If the same
    production reference was already processed, nothing is deducted.
    """
    reference = usage_reference(event_date)
    outcome: Dict[str, object] = {
        "reference": reference,
        "already_recorded": False,
        "deducted": [],
        "shortfalls": [],
        "skipped": [],
    }
    if usage_already_recorded(load_raw_usage(usage_path), reference):
        outcome["already_recorded"] = True
        return outcome

    raw = load_raw_inventory(raw_path)
    if raw.empty:
        return outcome
    today = pd.Timestamp(event_date).date()
    # Working copy of remaining quantities so successive items draw down stock.
    pool = {
        str(row["raw_id"]): float(row["remaining_quantity"] or 0)
        for _, row in raw.iterrows()
    }
    usage_rows: List[dict] = []
    for menu_item, servings in production_by_item.items():
        count = int(servings or 0)
        if count <= 0:
            continue
        recipe = recipe_for(menu_item)
        if not recipe:
            outcome["skipped"].append(f"{menu_item} (no recipe defined)")
            continue
        for ingredient, per_serving in recipe:
            needed = float(per_serving) * count
            if needed <= 0:
                continue
            candidates = usable_batches(raw, today)
            candidates = candidates[
                candidates["canonical_ingredient"].astype(str) == str(ingredient)
            ]
            for _, row in candidates.iterrows():
                if needed <= 1e-9:
                    break
                rid = str(row["raw_id"])
                available = pool.get(rid, 0.0)
                if available <= 0:
                    continue
                take = min(needed, available)
                pool[rid] = round(available - take, 6)
                needed -= take
                usage_rows.append(
                    {
                        "date": today.isoformat(),
                        "menu_item": menu_item,
                        "ingredient": ingredient,
                        "quantity_used": round(take, 6),
                        "unit": ingredient_unit(ingredient),
                        "production_reference": reference,
                        "raw_id": rid,
                    }
                )
                outcome["deducted"].append(
                    {
                        "raw_id": rid,
                        "ingredient": ingredient,
                        "quantity": round(take, 6),
                        "unit": ingredient_unit(ingredient),
                    }
                )
            if needed > 1e-9 and ingredient not in outcome["shortfalls"]:
                outcome["shortfalls"].append(ingredient)

    if not usage_rows:
        return outcome

    today_now = pd.Timestamp.today().date()
    updated = raw.copy()
    updated["remaining_quantity"] = pd.to_numeric(
        updated["remaining_quantity"], errors="coerce"
    ).fillna(0)
    for index, row in updated.iterrows():
        rid = str(row["raw_id"])
        if rid not in pool:
            continue
        new_remaining = max(round(pool[rid], 6), 0.0)
        updated.at[index, "remaining_quantity"] = new_remaining
        expiry = row["expiry_date"]
        if pd.notna(expiry):
            updated.at[index, "status"] = raw_status(new_remaining, expiry, today_now)
    save_raw_inventory(raw_path, updated)
    append_raw_usage(usage_path, usage_rows)
    return outcome


# ---------------------------------------------------------------------------
# Baseline preparedness
# ---------------------------------------------------------------------------

def baseline_demand_by_item(
    item_history, today: date, window_days: int, buffer_pct: float
) -> Dict[str, float]:
    """Recent average demand per canonical menu item plus a safety buffer.

    Uses ``actual_consumption`` from the last ``window_days`` completed days.
    The baseline never changes because of a single day's model fluctuation.
    """
    if item_history is None or item_history.empty:
        return {item: 0.0 for item in MENU_ITEMS}
    frame = item_history.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame[frame["date"].notna()]
    if frame.empty:
        return {item: 0.0 for item in MENU_ITEMS}
    latest = frame["date"].max().date()
    cutoff = pd.Timestamp(latest) - pd.Timedelta(days=int(window_days) - 1)
    window = frame[frame["date"].dt.date >= cutoff.date()]
    if window.empty:
        window = frame
    days = max(window["date"].dt.date.nunique(), 1)
    grouped = window.groupby("menu_item")["actual_consumption"].sum() / days
    factor = 1 + (float(buffer_pct) / 100)
    return {
        item: round(float(grouped.get(item, 0.0)) * factor, 1) for item in MENU_ITEMS
    }


def baseline_readiness(
    capacities: List[Dict[str, object]], baseline: Mapping[str, float]
) -> List[Dict[str, object]]:
    """Compare current capacity against the recommended baseline, per item."""
    rows = []
    for entry in capacities:
        item = entry["menu_item"]
        recommended = float(baseline.get(item, 0.0))
        current = int(entry["capacity"])
        gap = round(recommended - current, 1)
        rows.append(
            {
                "menu_item": item,
                "current_capacity": current,
                "recommended_baseline": recommended,
                "gap": gap,
                "status": "Above baseline" if gap <= 0 else "Below baseline",
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Usage analytics (read-only; reuses the existing usage history)
# ---------------------------------------------------------------------------

def usable_usage(usage, start_date=None, end_date=None) -> pd.DataFrame:
    """Clean usage rows for analytics.

    Drops rows that cannot be used (no ingredient, no menu item, non-numeric or
    non-positive quantity, missing unit, unparseable date, out of range) rather
    than letting one bad record break the whole page. Nothing is invented.
    """
    if usage is None or usage.empty:
        return pd.DataFrame(columns=RAW_USAGE_COLUMNS)
    frame = usage.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["ingredient"] = frame["ingredient"].fillna("").astype(str).str.strip()
    frame["menu_item"] = frame["menu_item"].fillna("").astype(str).str.strip()
    frame["unit"] = frame["unit"].fillna("").astype(str).str.strip()
    frame["quantity_used"] = pd.to_numeric(frame["quantity_used"], errors="coerce")
    frame = frame[
        frame["date"].notna()
        & (frame["ingredient"] != "")
        & (frame["menu_item"] != "")
        & (frame["unit"] != "")
        & frame["quantity_used"].notna()
        & (frame["quantity_used"] > 0)
    ]
    if start_date is not None:
        frame = frame[frame["date"].dt.date >= start_date]
    if end_date is not None:
        frame = frame[frame["date"].dt.date <= end_date]
    return frame.reset_index(drop=True)


def usage_totals_by_unit(usage) -> pd.DataFrame:
    """Total quantity per unit. Units are never summed together."""
    if usage.empty:
        return pd.DataFrame(columns=["Unit", "Total used", "Records"])
    grouped = (
        usage.groupby("unit", as_index=False)
        .agg(**{"Total used": ("quantity_used", "sum"), "Records": ("quantity_used", "size")})
        .rename(columns={"unit": "Unit"})
        .sort_values("Total used", ascending=False)
        .reset_index(drop=True)
    )
    return grouped


def usage_by_ingredient(usage) -> pd.DataFrame:
    """Ingredient x unit usage totals, largest first."""
    if usage.empty:
        return pd.DataFrame(columns=["Ingredient", "Unit", "Quantity used"])
    return (
        usage.groupby(["ingredient", "unit"], as_index=False)["quantity_used"]
        .sum()
        .rename(columns={"ingredient": "Ingredient", "unit": "Unit", "quantity_used": "Quantity used"})
        .sort_values("Quantity used", ascending=False)
        .reset_index(drop=True)
    )


def usage_by_menu_item(usage) -> pd.DataFrame:
    """Menu item x ingredient x unit usage totals from real usage records."""
    if usage.empty:
        return pd.DataFrame(columns=["Menu item", "Ingredient", "Unit", "Quantity used"])
    return (
        usage.groupby(["menu_item", "ingredient", "unit"], as_index=False)["quantity_used"]
        .sum()
        .rename(columns={
            "menu_item": "Menu item", "ingredient": "Ingredient",
            "unit": "Unit", "quantity_used": "Quantity used",
        })
        .sort_values(["Menu item", "Quantity used"], ascending=[True, False])
        .reset_index(drop=True)
    )


def usage_over_time(usage, unit: str) -> pd.DataFrame:
    """Daily usage totals for a single unit, so kg and L are never mixed."""
    if usage.empty:
        return pd.DataFrame(columns=["date", "Quantity used"])
    subset = usage[usage["unit"] == unit]
    if subset.empty:
        return pd.DataFrame(columns=["date", "Quantity used"])
    return (
        subset.groupby("date", as_index=False)["quantity_used"]
        .sum()
        .rename(columns={"quantity_used": "Quantity used"})
        .sort_values("date")
        .reset_index(drop=True)
    )
