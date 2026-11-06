"""Verify the saved California Housing model and its exported test results.

Run from the project root:
    python scripts/verify_artifacts.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / "Dataset" / "housing.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "Housing_Price_Prediction_System.pkl"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
PREDICTIONS_PATH = PROJECT_ROOT / "models" / "tuned_model_test_predictions.csv"


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def main() -> None:
    for path in (DATA_PATH, MODEL_PATH, METADATA_PATH, PREDICTIONS_PATH):
        if not path.is_file():
            fail(f"Required file is missing: {path.relative_to(PROJECT_ROOT)}")

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    dataset_hash = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    if dataset_hash != metadata["dataset_sha256"]:
        fail("Dataset SHA-256 does not match model metadata.")

    df = pd.read_csv(DATA_PATH)
    target = metadata["target"]
    if target not in df.columns:
        fail(f"Target column not found: {target}")

    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(metadata["test_size"]),
        random_state=int(metadata["random_state"]),
    )

    # Only load a pickle/joblib artifact from a source you trust.
    model = joblib.load(MODEL_PATH)
    predictions = model.predict(X_test)

    metrics = {
        "MAE": float(mean_absolute_error(y_test, predictions)),
        "MSE": float(mean_squared_error(y_test, predictions)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "R2 Score": float(r2_score(y_test, predictions)),
    }

    for metric_name, calculated in metrics.items():
        saved = float(metadata["test_metrics"][metric_name])
        if not np.isclose(calculated, saved, rtol=1e-8, atol=1e-6):
            fail(
                f"{metric_name} differs from metadata: "
                f"calculated={calculated}, saved={saved}"
            )

    exported = pd.read_csv(PREDICTIONS_PATH)
    if len(exported) != len(y_test):
        fail("Prediction CSV row count does not match the test split.")
    if not np.allclose(
        exported["Actual Price"].to_numpy(),
        y_test.to_numpy(),
        rtol=0,
        atol=1e-8,
    ):
        fail("Actual values in prediction CSV do not match the test split.")
    if not np.allclose(
        exported["Predicted Price"].to_numpy(),
        predictions,
        rtol=1e-8,
        atol=1e-6,
    ):
        fail("Predictions in CSV do not match the saved model output.")

    expected_version = metadata.get(
        "scikit_learn_version",
        metadata.get("python_scikit_learn_version", "unknown"),
    )
    chart_count = len(list((PROJECT_ROOT / "visualizations").glob("*.png")))

    print("PASS: Dataset SHA-256 matches model metadata.")
    print("PASS: Saved model loaded and predicted the held-out test split.")
    print("PASS: Recomputed metrics match model metadata.")
    print("PASS: Exported prediction CSV matches model predictions.")
    print(f"Training scikit-learn version: {expected_version}")
    print(f"Current scikit-learn version:  {sklearn.__version__}")
    if str(expected_version) != sklearn.__version__:
        print(
            "WARNING: scikit-learn versions differ. Install the exact version "
            "listed in requirements.txt before relying on the serialized model."
        )
    print(f"Visualization PNG files found: {chart_count}")
    print("Recomputed metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:,.6f}")


if __name__ == "__main__":
    main()
