"""Centralized kitchen configuration: canonical menu, raw ingredients and BOM.

Everything about the four-item menu and the seven raw ingredients lives here so
that dashboard.py and the raw-inventory subsystem never hardcode the menu.

NOTE: the recipe quantities and the baseline/buffer defaults below are
PROTOTYPE HACKATHON VALUES chosen to keep the demo coherent. They are not
food-safety, nutrition or industry standards.
"""

from typing import Dict, List, Tuple

# --- canonical menu -------------------------------------------------------

MENU_ITEMS: List[str] = [
    "Chai",
    "Rice with Chicken",
    "Rice with Tomato",
    "Rice with Dal",
]

# --- canonical raw ingredients -------------------------------------------

RAW_INGREDIENTS: List[str] = [
    "Tea Leaves",
    "Sugar",
    "Milk",
    "Rice",
    "Dal",
    "Tomato",
    "Chicken",
]

# Canonical unit per ingredient (weight-based -> kg, Milk -> litres).
INGREDIENT_UNITS: Dict[str, str] = {
    "Tea Leaves": "kg",
    "Sugar": "kg",
    "Milk": "L",
    "Rice": "kg",
    "Dal": "kg",
    "Tomato": "kg",
    "Chicken": "kg",
}

SUPPORTED_UNITS: List[str] = ["kg", "g", "L", "ml"]

# Deterministic unit conversion factors into the canonical unit above.
UNIT_TO_CANONICAL: Dict[Tuple[str, str], float] = {
    ("kg", "kg"): 1.0,
    ("g", "kg"): 0.001,
    ("L", "L"): 1.0,
    ("ml", "L"): 0.001,
}

# --- recipe / BOM ---------------------------------------------------------
# ingredient -> quantity per single serving, in the ingredient's canonical unit.
RECIPES: Dict[str, List[Tuple[str, float]]] = {
    "Chai": [("Tea Leaves", 0.005), ("Sugar", 0.010), ("Milk", 0.050)],
    "Rice with Chicken": [("Rice", 0.120), ("Chicken", 0.100)],
    "Rice with Tomato": [("Rice", 0.120), ("Tomato", 0.080)],
    "Rice with Dal": [("Rice", 0.120), ("Dal", 0.060)],
}

# --- expiry intelligence ---------------------------------------------------

# Centralized display thresholds (days remaining). These drive status wording
# and table colouring only; the underlying expiry dates are never changed.
EXPIRING_SOON_DAYS = 3
APPROACHING_EXPIRY_DAYS = 7

# Tone names used by the UI so colour never has to be inferred ad hoc.
TONE_EXPIRED = "expired"
TONE_EXPIRING = "expiring"
TONE_APPROACHING = "approaching"
TONE_HEALTHY = "healthy"
TONE_DEPLETED = "depleted"
TONE_UNKNOWN = "unknown"

# --- baseline preparedness -------------------------------------------------

BASELINE_WINDOW_DAYS = 7
BASELINE_SAFETY_BUFFER_PCT = 10.0

PROTOTYPE_NOTE = (
    "Recipe and baseline values are prototype demo settings, not food-safety, "
    "nutrition or industry standards."
)


def recipe_for(menu_item: str):
    """Return the BOM for a menu item, or None when the recipe is unknown."""
    return RECIPES.get(str(menu_item).strip())


def ingredient_unit(ingredient: str) -> str:
    """Canonical unit for an ingredient, defaulting to kg for unknown names."""
    return INGREDIENT_UNITS.get(ingredient, "kg")


def unit_supported(unit: str) -> bool:
    return str(unit).strip() in SUPPORTED_UNITS


"""Persisted kitchen settings with centralized defaults.

Defaults live in :mod:`src.kitchen_config`. This module only stores the
administrator's overrides, so there is a single source of truth for the
shipped values and a single place that validates what the admin changes.

Scope is deliberately narrow: it stores configuration numbers and boolean
notification preferences. It never touches inventory, usage history, surplus,
redistribution or the demand model.
"""

import json
from pathlib import Path
from typing import Dict

from src.kitchen_config import (
    APPROACHING_EXPIRY_DAYS,
    BASELINE_SAFETY_BUFFER_PCT,
    BASELINE_WINDOW_DAYS,
    EXPIRING_SOON_DAYS,
)

SETTINGS_FILENAME = "kitchen_settings.json"

NOTIFY_STOCK = "stock_alerts"
NOTIFY_EXPIRY = "expiry_alerts"
NOTIFY_PREDICTION = "prediction_alerts"

NOTIFICATION_KEYS = (NOTIFY_STOCK, NOTIFY_EXPIRY, NOTIFY_PREDICTION)

# Surplus shelf-life overrides, keyed by canonical menu item. An empty mapping
# means "use the existing surplus_shelf_life.csv values", so there is no
# duplicated shelf-life source of truth. When an item is present here it
# overrides the CSV value for newly created surplus batches only.
SURPLUS_SHELF_LIFE_KEY = "surplus_shelf_life_days"


def default_settings() -> Dict:
    """Shipped defaults, sourced from kitchen_config."""
    return {
        "baseline_window_days": int(BASELINE_WINDOW_DAYS),
        "safety_buffer_pct": float(BASELINE_SAFETY_BUFFER_PCT),
        "expiring_soon_days": int(EXPIRING_SOON_DAYS),
        "approaching_expiry_days": int(APPROACHING_EXPIRY_DAYS),
        NOTIFY_STOCK: True,
        NOTIFY_EXPIRY: True,
        NOTIFY_PREDICTION: True,
        SURPLUS_SHELF_LIFE_KEY: {},
    }


class SettingsError(ValueError):
    """Invalid settings; the saved configuration is left untouched."""


def validate_settings(candidate: Dict) -> Dict:
    """Validate and coerce a candidate, returning a clean settings dict.

    Raises SettingsError rather than writing anything, so a bad value can
    never corrupt the stored configuration.
    """
    base = default_settings()

    def as_int(name, minimum):
        try:
            value = int(candidate.get(name, base[name]))
        except (TypeError, ValueError):
            raise SettingsError(f"{name} must be a whole number.")
        if value < minimum:
            raise SettingsError(f"{name} must be at least {minimum}.")
        return value

    def as_float(name, minimum):
        try:
            value = float(candidate.get(name, base[name]))
        except (TypeError, ValueError):
            raise SettingsError(f"{name} must be a number.")
        if value != value or value < minimum:
            raise SettingsError(f"{name} must be at least {minimum}.")
        return value

    expiring = as_int("expiring_soon_days", 0)
    approaching = as_int("approaching_expiry_days", 0)
    if approaching < expiring:
        raise SettingsError(
            "Approaching expiry must be greater than or equal to Expiring soon."
        )

    # Surplus shelf life is a per-menu-item override map. Absent or empty means
    # "fall back to surplus_shelf_life.csv", so no value is duplicated here.
    shelf_raw = candidate.get(SURPLUS_SHELF_LIFE_KEY, base[SURPLUS_SHELF_LIFE_KEY])
    if shelf_raw is None:
        shelf_raw = {}
    if not isinstance(shelf_raw, dict):
        raise SettingsError("Surplus shelf life must be a set of menu item values.")
    shelf_life = {}
    for menu_item, value in shelf_raw.items():
        if menu_item not in MENU_ITEMS:
            raise SettingsError(f"'{menu_item}' is not a canonical menu item.")
        try:
            days = int(value)
        except (TypeError, ValueError):
            raise SettingsError(f"Shelf life for {menu_item} must be a whole number.")
        if days < 1:
            raise SettingsError(f"Shelf life for {menu_item} must be at least 1 day.")
        shelf_life[menu_item] = days

    return {
        "baseline_window_days": as_int("baseline_window_days", 1),
        "safety_buffer_pct": as_float("safety_buffer_pct", 0.0),
        "expiring_soon_days": expiring,
        "approaching_expiry_days": approaching,
        **{key: bool(candidate.get(key, base[key])) for key in NOTIFICATION_KEYS},
        SURPLUS_SHELF_LIFE_KEY: shelf_life,
    }


def load_settings(path: Path) -> Dict:
    """Load saved settings, falling back to defaults when absent or unreadable."""
    path = Path(path)
    if not path.exists():
        return default_settings()
    try:
        raw = json.loads(path.read_text(encoding="utf-8") or "{}")
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return default_settings()
    if not isinstance(raw, dict):
        return default_settings()
    try:
        return validate_settings(raw)
    except SettingsError:
        # A corrupted file must never break the app; fall back to defaults.
        return default_settings()


def save_settings(path: Path, candidate: Dict) -> Dict:
    """Validate then persist. Nothing is written when validation fails."""
    cleaned = validate_settings(candidate)
    Path(path).write_text(json.dumps(cleaned, indent=2), encoding="utf-8")
    return cleaned


def restore_defaults(path: Path) -> Dict:
    """Reset configuration only. No inventory, usage or model data is touched."""
    return save_settings(path, default_settings())


def menu_items_using(ingredient: str) -> List[str]:
    """Menu items whose recipe uses the given canonical ingredient."""
    return [
        item
        for item in MENU_ITEMS
        if any(name == ingredient for name, _ in RECIPES.get(item, []))
    ]

