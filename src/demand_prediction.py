from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


FEATURE_COLUMNS = [
    "day_of_week", "meal_type", "menu_type", "holiday", "special_event",
    "temperature", "rainfall",
]
TARGET_COLUMN = "actual_consumption"
CATEGORICAL_FEATURES = ["day_of_week", "meal_type", "menu_type"]
NUMERIC_FEATURES = ["holiday", "special_event", "temperature", "rainfall"]


def _features(data):
    features = data.copy()
    if "date" in features:
        dates = pd.to_datetime(features["date"])
        features["day_of_week"] = dates.dt.day_name()
    return features[FEATURE_COLUMNS]


def train_model(history):
    preprocessor = ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )
    model = Pipeline([("preprocessor", preprocessor), ("regressor", Ridge(alpha=1.0))])
    model.fit(_features(history), history[TARGET_COLUMN])
    return model


def train_or_load_model(history, model_path: Path):
    if model_path.exists():
        try:
            return joblib.load(model_path)
        except (EOFError, ValueError):
            model = train_model(history)
    else:
        model = train_model(history)
    joblib.dump(model, model_path)
    return model


def predict_demand(model, operation):
    prediction = model.predict(_features(operation))[0]
    return max(0, round(float(prediction)))
