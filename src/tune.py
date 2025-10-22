# src/tune.py
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import itertools
import mlflow

DATA_DIR = os.path.join("data", "processed")
OUT_BEST = os.path.join("models", "model_tuned.joblib")
EXPERIMENT_NAME = "credit_scoring_tuning"


def get_tracking_uri():
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

    # Grid values (at least two values each)
    n_estimators_list = [50, 100]
    max_depth_list = [6, 12]

    best_score = -1.0
    best_model = None
    best_params = None

    # Outer grouping run
    with mlflow.start_run(run_name="tuning_outer_run") as outer:
        mlflow.log_param("search", "grid")

        for n_est, m_depth in itertools.product(n_estimators_list, max_depth_list):
            with mlflow.start_run(nested=True) as inner:
                mlflow.log_param("n_estimators", n_est)
                mlflow.log_param("max_depth", m_depth)

                model = RandomForestClassifier(n_estimators=n_est, max_depth=m_depth, random_state=42)
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                probs = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None

                acc = float(accuracy_score(y_val, preds))
                auc = float(roc_auc_score(y_val, probs)) if probs is not None else float("nan")

                mlflow.log_metric("val_accuracy", acc)
                mlflow.log_metric("val_auc", auc)

                # Save per-run model artifact
                run_model_path = os.path.join("models", f"model_{n_est}_{m_depth}.joblib")
                os.makedirs(os.path.dirname(run_model_path), exist_ok=True)
                joblib.dump(model, run_model_path)
                mlflow.log_artifact(run_model_path, artifact_path="models")

                if acc > best_score:
                    best_score = acc
                    best_model = model
                    best_params = {"n_estimators": n_est, "max_depth": m_depth}

        # After nested runs: log best
        mlflow.log_metric("best_val_accuracy", float(best_score))
        mlflow.log_param("best_params", str(best_params))

        if best_model is not None:
            os.makedirs(os.path.dirname(OUT_BEST), exist_ok=True)
            joblib.dump(best_model, OUT_BEST)
            mlflow.log_artifact(OUT_BEST, artifact_path="models")

    print("Tuning finished. Best:", best_params, "score:", best_score)
    print(f"Best model saved to {OUT_BEST}")


if __name__ == "__main__":
    main()