# src/preprocess.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

RAW = os.path.join("data", "raw", "credit_scoring.csv")
OUT_DIR = os.path.join("data", "processed")
SCALER_PATH = os.path.join("models", "scaler.joblib")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

TARGET = "Default"  # label column in your dataset
ID_COL = "ID"


def main():
    print("Loading raw data from:", RAW)
    df = pd.read_csv(RAW)
    print("Raw shape:", df.shape)

    # Drop ID column
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])

    # Ensure target exists
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in data columns: {df.columns.tolist()}")

    # Separate features/target
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Basic cleaning: numeric imputation
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # Replace infs if any
    X[numeric_cols] = X[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # Impute numeric with median
    imputer = SimpleImputer(strategy="median")
    X[numeric_cols] = imputer.fit_transform(X[numeric_cols])

    # If any non-numeric columns exist, do one-hot encoding
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if len(cat_cols) > 0:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # Reassemble cleaned dataframe
    df_clean = pd.concat([X, y.reset_index(drop=True)], axis=1)

    # Stratified split: train/val/test (60/20/20)
    stratify_col = y if len(y.unique()) > 1 else None
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_col
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval if len(y_trainval.unique())>1 else None
    )
    # Note: second split 0.25 of 0.8 => 0.2, giving 0.6/0.2/0.2

    # Scale numeric features with StandardScaler (fit on train)
    scaler = StandardScaler()
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    # Fit scaler on train numeric columns and transform all three splits
    X_train_num = scaler.fit_transform(X_train[num_cols])
    X_val_num = scaler.transform(X_val[num_cols])
    X_test_num = scaler.transform(X_test[num_cols])

    # Replace numeric columns with scaled values
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[num_cols] = X_train_num
    X_val_scaled[num_cols] = X_val_num
    X_test_scaled[num_cols] = X_test_num

    # Save CSVs (features + target as last column)
    train_df = X_train_scaled.copy()
    train_df[TARGET] = y_train.values
    val_df = X_val_scaled.copy()
    val_df[TARGET] = y_val.values
    test_df = X_test_scaled.copy()
    test_df[TARGET] = y_test.values

    train_df.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(OUT_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(OUT_DIR, "test.csv"), index=False)

    # Save scaler & imputer for reproducibility
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(imputer, os.path.join("models", "imputer.joblib"))

    print("Preprocessing complete. Outputs:")
    print(" ", os.path.join(OUT_DIR, "train.csv"))
    print(" ", os.path.join(OUT_DIR, "val.csv"))
    print(" ", os.path.join(OUT_DIR, "test.csv"))
    print(" ", SCALER_PATH)


if __name__ == "__main__":
    main()