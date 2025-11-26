import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

LABEL_CANDS = ["label", "malicious"]

def load_csv(path):
    df = pd.read_csv(path)
    label = None
    for c in LABEL_CANDS:
        possible_labels = ["label", "Label", "malicious", "y", "class", "target"]

        label_col = None
        for col in possible_labels:
            if col in df.columns:
                label_col = col
                break

            if label_col is None:
                raise ValueError(f"No label column found. Columns available: {df.columns}")

    y = df[label].astype(int).values
    X = df.drop(columns=[label]).select_dtypes(include=[np.number]).fillna(0).astype(np.float32)
    return X.values, y

def prepare(train_csv, test_csv=None, val_split=0.2, seed=42, scaler_out=None):
    X, y = load_csv(train_csv)
    if test_csv:
        X_val, y_val = load_csv(test_csv)
        X_tr, y_tr = X, y
    else:
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=val_split, random_state=seed, stratify=y)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)
    if scaler_out:
        joblib.dump(scaler, scaler_out)
    x_min = X_tr.min(axis=0).astype(np.float32)
    x_max = X_tr.max(axis=0).astype(np.float32)
    return (X_tr.astype(np.float32), y_tr.astype(np.float32),
            X_val.astype(np.float32), y_val.astype(np.float32),
            scaler, x_min, x_max)