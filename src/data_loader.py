import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# All supported label names (case-insensitive)
POSSIBLE_LABELS = ["label", "malicious", "y", "class", "target"]

def load_csv(path):
    df = pd.read_csv(path)

    # Normalize column names: remove spaces & lowercase
    df.columns = df.columns.str.strip().str.lower()

    # Find label column
    label_col = None
    for col in df.columns:
        if col in POSSIBLE_LABELS:
            label_col = col
            break

    if label_col is None:
        raise ValueError(
            f"No label column found after normalization. Columns available: {df.columns}"
        )

    # Extract target
    y = df[label_col].astype(int).values

    # Extract numeric features only
    X = df.drop(columns=[label_col]).select_dtypes(include=[np.number]).fillna(0).astype(np.float32)

    return X.values, y


def prepare(train_csv, test_csv=None, val_split=0.2, seed=42, scaler_out=None):
    # Load training data
    X, y = load_csv(train_csv)

    # Test data is separate file
    if test_csv:
        X_val, y_val = load_csv(test_csv)
        X_tr, y_tr = X, y
    else:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=val_split, random_state=seed, stratify=y
        )

    # Scale
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)

    if scaler_out:
        joblib.dump(scaler, scaler_out)

    # Compute feature-wise min/max
    x_min = X_tr.min(axis=0).astype(np.float32)
    x_max = X_tr.max(axis=0).astype(np.float32)

    return (
        X_tr.astype(np.float32),
        y_tr.astype(np.float32),
        X_val.astype(np.float32),
        y_val.astype(np.float32),
        scaler,
        x_min,
        x_max,
    )
