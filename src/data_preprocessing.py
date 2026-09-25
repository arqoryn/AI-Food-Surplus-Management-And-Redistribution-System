from pathlib import Path

import pandas as pd


HISTORY_COLUMNS = [
    "date", "day_of_week", "meal_type", "menu_type", "planned_quantity",
    "actual_consumption", "waste_quantity", "holiday", "special_event",
    "temperature", "rainfall",
]
DAILY_COLUMNS = ["date", "meal_type", "menu_type", "planned_quantity", "holiday", "special_event"]
RECIPIENT_COLUMNS = [
    "recipient_id", "name", "type", "latitude", "longitude", "capacity",
    "current_need", "priority", "distance_km", "active",
]
RECORD_COLUMNS = ["date", "recipient_id", "allocated_quantity", "distance_km", "status"]


def validate_columns(data, required_columns, path):
    missing = sorted(set(required_columns) - set(data.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def load_csv(path: Path, required_columns):
    data = pd.read_csv(path)
    validate_columns(data, required_columns, path)
    return data


def save_redistribution_records(path: Path, new_records):
    validate_columns(new_records, RECORD_COLUMNS, "new redistribution records")
    existing = load_csv(path, RECORD_COLUMNS)
    combined = pd.concat([existing, new_records[RECORD_COLUMNS]], ignore_index=True)
    combined.to_csv(path, index=False)
