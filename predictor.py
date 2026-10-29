from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


APP_DIR = Path(__file__).resolve().parent
FEATURE_NAME = "raw_marks"
MIN_RAW_MARKS = 0
MAX_RAW_MARKS = 300
DEFAULT_RAW_MARKS = 160
PERCENTILE_MODEL_FILE = "percentile_model (1).pkl"
RANK_MODEL_FILE = "rank_model (1).pkl"


def validate_raw_marks(raw_marks: Any) -> int:
    if isinstance(raw_marks, bool) or not isinstance(raw_marks, (int, np.integer)):
        raise ValueError("Raw marks must be a whole number.")
    if not MIN_RAW_MARKS <= raw_marks <= MAX_RAW_MARKS:
        raise ValueError(
            f"Raw marks must be between {MIN_RAW_MARKS} and {MAX_RAW_MARKS}."
        )
    return int(raw_marks)


def _validate_model_schema(model: Any, file_name: str) -> None:
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is not None and tuple(feature_names) != (FEATURE_NAME,):
        raise RuntimeError(
            f"{file_name} expects {list(feature_names)}, not [{FEATURE_NAME!r}]."
        )

    feature_count = getattr(model, "n_features_in_", None)
    if feature_count is not None and feature_count != 1:
        raise RuntimeError(f"{file_name} expects {feature_count} input features.")


def _load_models(model_dir: Path) -> tuple[Any, Any]:
    loaded_models = []
    for file_name in (PERCENTILE_MODEL_FILE, RANK_MODEL_FILE):
        model_path = model_dir / file_name
        if not model_path.is_file():
            raise FileNotFoundError(f"Required model file is missing: {file_name}")
        try:
            model = joblib.load(model_path)
        except Exception as error:
            raise RuntimeError(
                f"Could not load {file_name}. Check the installed model dependencies."
            ) from error
        _validate_model_schema(model, file_name)
        loaded_models.append(model)
    return loaded_models[0], loaded_models[1]


def predict_scores(
    raw_marks: Any, percentile_model: Any, rank_model: Any
) -> tuple[float, float]:
    marks = validate_raw_marks(raw_marks)
    model_input = pd.DataFrame({FEATURE_NAME: pd.Series([marks], dtype="int64")})
    percentile = float(percentile_model.predict(model_input)[0])
    rank = float(rank_model.predict(model_input)[0])
    if not np.isfinite(percentile) or not np.isfinite(rank):
        raise ValueError("The models returned a non-finite prediction.")
    return percentile, rank