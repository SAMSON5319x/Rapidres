import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Supported label names (case-insensitive)
POSSIBLE_LABELS = ["label", "malicious", "y", "class", "target"]

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    return df

def _find_label_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        if col in POSSIBLE_LABELS:
            return col
    raise ValueError(f"No label column found. Available columns: {list(df.columns)}")

def _normalize_labels(y_raw: pd.Series) -> np.ndarray:
    # Handle strings / objects
    if y_raw.dtype.kind in "OUSb":  # object, unicode, string, bool
        y = y_raw.astype(str).str.lower().map({
            "1": 1, "true": 1, "malicious": 1, "malware": 1, "pos": 1, "positive": 1,
            "0": 0, "false": 0, "benign": 0, "neg": 0, "negative": 0
        })
        y = y.fillna(0).astype(np.float32)
    else:
        y = pd.to_numeric(y_raw, errors="coerce").fillna(0).astype(np.float32)
        uniq = set(np.unique(y))
        if uniq == {-1.0, 1.0}:              # {-1,1} -> {0,1}
            y = (y > 0).astype(np.float32)
        elif uniq == {1.0, 2.0}:             # {1,2} -> {0,1}
            y = (y - 1.0).astype(np.float32)
        elif max(uniq) > 1.0 or min(uniq) < 0.0:
            # Fallback: everything > 0 is positive
            y = (y > 0).astype(np.float32)
        else:
            y = np.clip(y, 0.0, 1.0).astype(np.float32)
    return y

def _numeric_features(df: pd.DataFrame, drop_cols) -> pd.DataFrame:
    X = df.drop(columns=drop_cols, errors="ignore")
    X = X.select_dtypes(include=[np.number]).copy()
    # Clean infinities/NaNs
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X.astype(np.float32)

def load_csv(path: str) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(path)
    df = _normalize_columns(df)
    label_col = _find_label_column(df)
    y = _normalize_labels(df[label_col])
    X = _numeric_features(df, drop_cols=[label_col])
    return X, y

def _align_columns(X_train_df: pd.DataFrame, X_test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reindex test to train's columns; unseen columns in test are dropped,
    missing columns are filled with zeros.
    """
    X_test_df = X_test_df.copy()
    # Add missing columns as zeros
    missing = [c for c in X_train_df.columns if c not in X_test_df.columns]
    for c in missing:
        X_test_df[c] = 0.0
    # Drop extras not present in train
    X_test_df = X_test_df.reindex(columns=X_train_df.columns, fill_value=0.0)
    return X_test_df

def prepare(train_csv, test_csv=None, val_split=0.2, seed=42, scaler_out=None):
    # Load training
    X_train_df, y_train = load_csv(train_csv)

    if test_csv:
        X_val_df, y_val = load_csv(test_csv)
        # Align columns between train and test
        X_val_df = _align_columns(X_train_df, X_val_df)
        X_tr_df, y_tr = X_train_df, y_train
        X_va_df, y_va = X_val_df, y_val
    else:
        X_tr_df, X_va_df, y_tr, y_va = train_test_split(
            X_train_df, y_train, test_size=val_split, random_state=seed, stratify=y_train
        )

    # Standardize
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr_df.values).astype(np.float32)
    X_va = scaler.transform(X_va_df.values).astype(np.float32)

    if scaler_out:
        joblib.dump(scaler, scaler_out)

    # Feature-wise bounds in standardized space
    x_min = X_tr.min(axis=0).astype(np.float32)
    x_max = X_tr.max(axis=0).astype(np.float32)

    return (
        X_tr,
        y_tr.astype(np.float32),
        X_va,
        y_va.astype(np.float32),
        scaler,
        x_min,
        x_max,
    )
