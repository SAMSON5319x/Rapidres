import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def load_ember_csv(path):
    # path -> CSV or parquet containing features and a 'label' or 'malicious' column
    df = pd.read_csv(path)
    # EMBER labels often: 1 = malicious, 0 = benign
    if 'label' in df.columns:
        y = df['label'].values
        X = df.drop(columns=['label'])
    elif 'malicious' in df.columns:
        y = df['malicious'].values
        X = df.drop(columns=['malicious'])
    else:
        # adjust as needed
        raise ValueError("Label column not found.")
    return X, y

def prepare_data(train_csv, test_csv=None, test_size=0.2, random_state=42, save_scaler=None):
    X, y = load_ember_csv(train_csv)
    if test_csv is not None:
        X_test, y_test = load_ember_csv(test_csv)
        X_train, y_train = X, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y)
    # Some EMBER features are sparse; make sure to convert to numeric matrix
    X_train = X_train.fillna(0).astype(np.float32)
    X_test = X_test.fillna(0).astype(np.float32)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if save_scaler:
        joblib.dump(scaler, save_scaler)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler