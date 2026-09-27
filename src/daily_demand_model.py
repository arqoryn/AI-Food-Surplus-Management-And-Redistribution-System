import hashlib
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DAILY_FEATURE_COLUMNS = [
    "day_of_week",
    "holiday",
    "special_event",
    "weather",
    "precipitation",
]
DAILY_TARGET_COLUMN = "actual_consumption"
DAILY_MODEL_SCHEMA_VERSION = 1
DAILY_CATEGORICAL_FEATURES = ["day_of_week", "weather", "precipitation"]
DAILY_NUMERIC_FEATURES = ["holiday", "special_event"]


def _daily_features(data: pd.DataFrame) -> pd.DataFrame:
    features = data.copy()
    if "date" in features:
        dates = pd.to_datetime(features["date"])
        features["day_of_week"] = dates.dt.day_name()
    for column in DAILY_NUMERIC_FEATURES:
        features[column] = pd.to_numeric(features[column], errors="raise").astype(int)
    return features[DAILY_FEATURE_COLUMNS]


def _training_data_fingerprint(history: pd.DataFrame) -> str:
    training_data = history[
        ["date", "holiday", "special_event", "weather", "precipitation", DAILY_TARGET_COLUMN]
    ].copy()
    hashed_data = pd.util.hash_pandas_object(training_data, index=True).values.tobytes()
    return hashlib.sha256(hashed_data).hexdigest()


def train_daily_model(daily_history: pd.DataFrame) -> Pipeline:
    if daily_history.empty:
        raise ValueError("Cannot train daily demand model with empty daily history.")

    preprocessor = ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), DAILY_CATEGORICAL_FEATURES),
            ("numeric", "passthrough", DAILY_NUMERIC_FEATURES),
        ]
    )
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", Ridge(alpha=1.0)),
    ])

    X = _daily_features(daily_history)
    y = daily_history[DAILY_TARGET_COLUMN].astype(float)
    model.fit(X, y)

    model.model_schema_version = DAILY_MODEL_SCHEMA_VERSION
    model.model_feature_columns = tuple(DAILY_FEATURE_COLUMNS)
    model.training_data_fingerprint = _training_data_fingerprint(daily_history)
    return model


def train_or_load_daily_model(daily_history: pd.DataFrame, model_path: Path) -> Pipeline:
    if model_path.exists():
        try:
            model = joblib.load(model_path)
            if (
                getattr(model, "model_schema_version", None) == DAILY_MODEL_SCHEMA_VERSION
                and getattr(model, "training_data_fingerprint", None) == _training_data_fingerprint(daily_history)
            ):
                return model
        except (EOFError, ValueError, AttributeError, ModuleNotFoundError):
            pass

    model = train_daily_model(daily_history)
    joblib.dump(model, model_path)
    return model




# ---------------------------------------------------------------------------
# Item Allocation Foundation
# ---------------------------------------------------------------------------

def calculate_item_shares(item_history: pd.DataFrame) -> Dict[str, float]:
    """Calculate normalized historical consumption shares per menu item.

    item_share = sum(item actual_consumption) / sum(total actual_consumption)
    Shares strictly sum to 1.0.
    """
    if item_history.empty:
        raise ValueError("Cannot calculate item shares from empty item history.")

    grouped = item_history.groupby("menu_item")["actual_consumption"].sum()
    total_consumption = grouped.sum()

    if total_consumption <= 0:
        raise ValueError("Total historical item consumption must be greater than zero.")

    raw_shares = (grouped / total_consumption).to_dict()
    total_share = sum(raw_shares.values())
    normalized_shares = {k: float(v / total_share) for k, v in raw_shares.items()}
    return normalized_shares


def allocate_daily_to_items(
    daily_total: int,
    item_shares: Dict[str, float],
) -> Dict[str, int]:
    """Allocate daily integer total to menu items using deterministic largest-remainder method.

    Guarantees:
    - Exactly sum(allocated) == daily_total
    - Deterministic rounding (tie-broken alphabetically by item name)
    - All item quantities are non-negative integers
    """
    if daily_total < 0:
        raise ValueError("daily_total must be non-negative.")
    if not item_shares:
        raise ValueError("item_shares cannot be empty.")

    items = sorted(item_shares.keys())
    total_share = sum(item_shares[item] for item in items)
    if not np.isclose(total_share, 1.0, atol=1e-5):
        raise ValueError(f"item_shares must sum to 1.0, got {total_share}")

    exact_allocations = {item: daily_total * (item_shares[item] / total_share) for item in items}
    integer_allocations = {item: int(np.floor(exact_allocations[item])) for item in items}
    remainder = daily_total - sum(integer_allocations.values())

    fractional_parts = [
        (exact_allocations[item] - integer_allocations[item], item)
        for item in items
    ]
    fractional_parts.sort(key=lambda x: (-x[0], x[1]))

    for i in range(remainder):
        _, item = fractional_parts[i]
        integer_allocations[item] += 1

    return integer_allocations

def predict_daily_demand(model: Pipeline, operation_context: pd.DataFrame) -> int:
    prediction = model.predict(_daily_features(operation_context))[0]
    return max(0, round(float(prediction)))
