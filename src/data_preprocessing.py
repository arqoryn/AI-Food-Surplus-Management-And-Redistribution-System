from pathlib import Path

import pandas as pd


HISTORY_COLUMNS = [
    "date", "meal_type", "menu_type", "predicted_demand", "actual_production",
    "actual_consumption", "surplus_quantity", "holiday", "special_event",
    "weather", "precipitation",
]
DAILY_COLUMNS = [
    "date", "meal_type", "menu_type", "planned_quantity", "holiday", "special_event",
    "weather", "precipitation", "safety_buffer",
]
DAILY_HISTORY_COLUMNS = [
    "date", "predicted_demand", "actual_production", "actual_consumption",
    "surplus_quantity", "holiday", "special_event", "weather", "precipitation",
]
DAILY_OPERATIONS_COLUMNS = [
    "date", "planned_quantity", "holiday", "special_event",
    "weather", "precipitation", "safety_buffer",
]
ITEM_HISTORY_COLUMNS = [
    "date", "menu_item", "predicted_quantity", "actual_production",
    "actual_consumption", "surplus_quantity",
]
INVENTORY_COLUMNS = [
    "inventory_id", "date_added", "menu_item", "original_quantity",
    "remaining_quantity", "expiry_date", "status",
]
SHELF_LIFE_COLUMNS = ["menu_item", "shelf_life_days"]

RECIPIENT_COLUMNS = [
    "recipient_id", "name", "type", "latitude", "longitude", "capacity",
    "current_need", "priority", "distance_km", "active",
]
RECORD_COLUMNS = ["date", "recipient_id", "allocated_quantity", "distance_km", "status", "menu_item"]

WEATHER_OPTIONS = ["Sunny", "Cloudy", "Rainy", "Snowy"]
PRECIPITATION_OPTIONS = {
    "Sunny": ["No precipitation"],
    "Cloudy": ["No Rain", "Light Rain", "Heavy Rain"],
    "Rainy": ["Light Rain", "Heavy Rain"],
    "Snowy": ["Light Snow", "Heavy Snow"],
}


def validate_columns(data, required_columns, path):
    missing = sorted(set(required_columns) - set(data.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def validate_weather_precipitation(data, path):
    invalid = data[
        ~data.apply(
            lambda row: row["weather"] in WEATHER_OPTIONS
            and row["precipitation"] in PRECIPITATION_OPTIONS.get(row["weather"], []),
            axis=1,
        )
    ]
    if not invalid.empty:
        raise ValueError(f"{path} contains invalid weather/precipitation combinations")


def validate_history_values(data, path):
    numeric_columns = [
        "predicted_demand", "actual_production", "actual_consumption", "surplus_quantity",
    ]
    numeric_data = data[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if numeric_data.isna().any().any() or (numeric_data < 0).any().any():
        raise ValueError(f"{path} contains missing or negative outcome values")
    if (numeric_data["actual_consumption"] > numeric_data["actual_production"]).any():
        raise ValueError(f"{path} contains consumption greater than production")
    expected_surplus = numeric_data["actual_production"] - numeric_data["actual_consumption"]
    if not expected_surplus.eq(numeric_data["surplus_quantity"]).all():
        raise ValueError(f"{path} contains surplus values inconsistent with production and consumption")


def load_csv(path: Path, required_columns):
    data = pd.read_csv(path)
    data.columns = data.columns.str.replace("\ufeff", "", regex=False).str.strip()
    validate_columns(data, required_columns, path)
    if "weather" in required_columns and "precipitation" in required_columns:
        validate_weather_precipitation(data, path)
    if required_columns in (HISTORY_COLUMNS, DAILY_HISTORY_COLUMNS):
        validate_history_values(data, path)
    return data[required_columns]


def load_redistribution_records(path: Path):
    """Load redistribution records, tolerating legacy files without menu_item.

    Legacy rows keep all of their original values; they simply get an empty
    menu_item because they predate item-level redistribution.
    """
    data = pd.read_csv(path)
    data.columns = data.columns.str.replace("﻿", "", regex=False).str.strip()
    if "menu_item" not in data.columns:
        data["menu_item"] = ""
    validate_columns(data, RECORD_COLUMNS, path)
    return data[RECORD_COLUMNS]


def save_redistribution_records(path: Path, new_records):
    validate_columns(new_records, RECORD_COLUMNS, "new redistribution records")
    if new_records["menu_item"].isna().any() or (
        new_records["menu_item"].astype(str).str.strip() == ""
    ).any():
        raise ValueError("new redistribution records must include a menu_item")
    existing = load_redistribution_records(path)
    combined = pd.concat([existing, new_records[RECORD_COLUMNS]], ignore_index=True)
    combined.to_csv(path, index=False)


def append_kitchen_history(path: Path, new_record):
    validate_columns(new_record, HISTORY_COLUMNS, "new kitchen history record")
    if new_record.empty:
        raise ValueError("new kitchen history record is empty")
    validate_weather_precipitation(new_record, "new kitchen history record")
    validate_history_values(new_record, "new kitchen history record")

    existing = load_csv(path, HISTORY_COLUMNS)
    identity_columns = ["date", "meal_type", "menu_type"]
    existing_identity = existing[identity_columns].copy()
    new_identity = new_record[identity_columns].copy()
    existing_identity["date"] = pd.to_datetime(existing_identity["date"], errors="raise").dt.date.astype(str)
    new_identity["date"] = pd.to_datetime(new_identity["date"], errors="raise").dt.date.astype(str)
    existing_keys = set(existing_identity.astype(str).agg("|".join, axis=1))
    new_keys = new_identity.astype(str).agg("|".join, axis=1)
    if new_keys.duplicated().any() or any(key in existing_keys for key in new_keys):
        raise ValueError("kitchen history already contains an observation with the same date, meal type, and menu type")

    combined = pd.concat([existing, new_record[HISTORY_COLUMNS]], ignore_index=True)
    combined.to_csv(path, index=False)


def append_daily_history(path: Path, new_record):
    validate_columns(new_record, DAILY_HISTORY_COLUMNS, "new daily history record")
    if new_record.empty:
        raise ValueError("new daily history record is empty")
    validate_weather_precipitation(new_record, "new daily history record")
    validate_history_values(new_record, "new daily history record")

    existing = load_csv(path, DAILY_HISTORY_COLUMNS)
    existing_dates = set(pd.to_datetime(existing["date"], errors="raise").dt.date.astype(str))
    new_dates = pd.to_datetime(new_record["date"], errors="raise").dt.date.astype(str)
    if new_dates.duplicated().any() or any(d in existing_dates for d in new_dates):
        raise ValueError("daily history already contains an observation for the specified date")

    combined = pd.concat([existing, new_record[DAILY_HISTORY_COLUMNS]], ignore_index=True)
    combined.to_csv(path, index=False)


def validate_item_history_values(data, path):
    numeric_columns = [
        "predicted_quantity", "actual_production", "actual_consumption", "surplus_quantity",
    ]
    numeric_data = data[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if numeric_data.isna().any().any() or (numeric_data < 0).any().any():
        raise ValueError(f"{path} contains missing or negative item values")
    if (numeric_data["actual_consumption"] > numeric_data["actual_production"]).any():
        raise ValueError(f"{path} contains item consumption greater than item production")
    expected_surplus = numeric_data["actual_production"] - numeric_data["actual_consumption"]
    if not expected_surplus.eq(numeric_data["surplus_quantity"]).all():
        raise ValueError(f"{path} contains item surplus values inconsistent with production and consumption")


def append_item_history(path: Path, new_records):
    """Append one completed day's item-level records as a single batch.

    The whole batch is validated first and the date must not exist yet, so a day
    can only be written once and never partially.
    """
    validate_columns(new_records, ITEM_HISTORY_COLUMNS, "new item history records")
    if new_records.empty:
        raise ValueError("new item history records are empty")
    validate_item_history_values(new_records, "new item history records")

    new_identity = new_records[["date", "menu_item"]].astype(str)
    if new_identity.duplicated().any():
        raise ValueError("new item history records contain duplicate menu items for the same date")

    existing = load_csv(path, ITEM_HISTORY_COLUMNS)
    existing_dates = set(pd.to_datetime(existing["date"], errors="raise").dt.date.astype(str))
    new_dates = pd.to_datetime(new_records["date"], errors="raise").dt.date.astype(str)
    if any(date in existing_dates for date in new_dates.unique()):
        raise ValueError("item history already contains records for the specified date")

    combined = pd.concat([existing, new_records[ITEM_HISTORY_COLUMNS]], ignore_index=True)
    combined.to_csv(path, index=False)


def validate_inventory_values(data, path):
    numeric_columns = ["original_quantity", "remaining_quantity"]
    numeric_data = data[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if numeric_data.isna().any().any() or (numeric_data < 0).any().any():
        raise ValueError(f"{path} contains missing or negative inventory quantities")
    if (numeric_data["remaining_quantity"] > numeric_data["original_quantity"]).any():
        raise ValueError(f"{path} contains remaining quantity greater than original quantity")


def append_inventory(path: Path, new_records):
    """Append new surplus inventory batches, rejecting duplicate batches."""
    validate_columns(new_records, INVENTORY_COLUMNS, "new inventory records")
    if new_records.empty:
        raise ValueError("new inventory records are empty")
    validate_inventory_values(new_records, "new inventory records")

    if new_records["inventory_id"].astype(str).duplicated().any():
        raise ValueError("new inventory records contain duplicate inventory ids")
    new_identity = new_records[["date_added", "menu_item"]].astype(str)
    if new_identity.duplicated().any():
        raise ValueError("new inventory records contain duplicate batches for the same date and menu item")

    existing = load_csv(path, INVENTORY_COLUMNS)
    existing_ids = set(existing["inventory_id"].astype(str))
    if any(inventory_id in existing_ids for inventory_id in new_records["inventory_id"].astype(str)):
        raise ValueError("inventory already contains one of the provided inventory ids")
    existing_identity = existing[["date_added", "menu_item"]].astype(str)
    existing_keys = set(existing_identity.agg("|".join, axis=1))
    if any(key in existing_keys for key in new_identity.agg("|".join, axis=1)):
        raise ValueError("inventory already contains a batch for the specified date and menu item")

    combined = pd.concat([existing, new_records[INVENTORY_COLUMNS]], ignore_index=True)
    combined.to_csv(path, index=False)

