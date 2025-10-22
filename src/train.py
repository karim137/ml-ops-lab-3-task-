# src/train.py
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import mlflow
import numpy as np

DATA_DIR = os.path.join("data", "processed")
MODEL_PATH = os.path.join("models", "model_baseline.joblib")
EXPERIMENT_NAME = "credit_scoring_baseline"


def get_tracking_uri():
    # prefer env var MLFLOW_TRACKING_URI if set
    return os.environ.get("MLFLOW_TRACKING_URI")


def load_csv(path):
    return pd.read_csv(path)


def main():
    tracking_uri = get_tracking_uri()
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    train = load_csv(os.path.join(DATA_DIR, "train.csv"))
    val = load_csv(os.path.join(DATA_DIR, "val.csv"))

    feature_cols = train.columns.drop("Default")
    target_col = "Default"

    X_train = train[feature_cols].values
    y_train = train[target_col].values
    X_val = val[feature_cols].values
    y_val = val[target_col].values

    # Baseline model
    model = RandomForestClassifier(random_state=42, n_estimators=100)

    with mlflow.start_run(run_name="baseline_run"):
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None

        acc = float(accuracy_score(y_val, preds))
        auc = float(roc_auc_score(y_val, probs)) if probs is not None else float("nan")

        mlflow.log_param("model", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("val_accuracy", acc)
        mlflow.log_metric("val_auc", auc)

        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        mlflow.log_artifact(MODEL_PATH, artifact_path="models")

    print(f"Saved baseline model to {MODEL_PATH}. val_acc={acc:.4f}, val_auc={auc:.4f}")


if __name__ == "__main__":
    main()