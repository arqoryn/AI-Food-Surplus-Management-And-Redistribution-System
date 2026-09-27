import hashlib
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


FEATURE_COLUMNS = [
    "day_of_week", "meal_type", "menu_type", "holiday", "special_event",
    "weather", "precipitation",
]
TARGET_COLUMN = "actual_consumption"
MODEL_SCHEMA_VERSION = 3
TRAINING_DATA_COLUMNS = [
    "date", "meal_type", "menu_type", "holiday", "special_event",
    "weather", "precipitation", "actual_consumption",
]
CATEGORICAL_FEATURES = ["day_of_week", "meal_type", "menu_type", "weather", "precipitation"]
NUMERIC_FEATURES = ["holiday", "special_event"]


def _training_data_fingerprint(history):
    training_data = history[TRAINING_DATA_COLUMNS].copy()
    hashed_data = pd.util.hash_pandas_object(training_data, index=True).values.tobytes()
    return hashlib.sha256(hashed_data).hexdigest()


def _features(data):
    features = data.copy()
    if "date" in features:
        dates = pd.to_datetime(features["date"])
        features["day_of_week"] = dates.dt.day_name()
    for column in NUMERIC_FEATURES:
        features[column] = pd.to_numeric(features[column], errors="raise").astype(int)
    return features[FEATURE_COLUMNS]


def train_model(history):
    if history.empty:
        raise ValueError("cannot train demand model with an empty history dataset")
    preprocessor = ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )
    model = Pipeline([("preprocessor", preprocessor), ("regressor", Ridge(alpha=1.0))])
    model.fit(_features(history), history[TARGET_COLUMN])
    model.model_schema_version = MODEL_SCHEMA_VERSION
    model.model_feature_columns = tuple(FEATURE_COLUMNS)
    model.training_data_fingerprint = _training_data_fingerprint(history)
    return model


def train_or_load_model(history, model_path: Path):
    if model_path.exists():
        try:
            model = joblib.load(model_path)
            if (
                getattr(model, "model_schema_version", None) == MODEL_SCHEMA_VERSION
                and getattr(model, "training_data_fingerprint", None) == _training_data_fingerprint(history)
            ):
                return model
        except (EOFError, ValueError, AttributeError, ModuleNotFoundError):
            pass
        model = train_model(history)
    else:
        model = train_model(history)
    joblib.dump(model, model_path)
    return model


def predict_demand(model, operation):
    prediction = model.predict(_features(operation))[0]
    return max(0, round(float(prediction)))
