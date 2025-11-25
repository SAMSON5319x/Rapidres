import pandas as pd
import os

# Path to your parquet file
parquet_path = "archive/train_ember_2018_v2_features.parquet"
csv_path = "ember_train.csv"

print(f"Reading {parquet_path} ...")
df = pd.read_parquet(parquet_path)
print(f"Shape: {df.shape}")

# Save as CSV (with compression off for faster load)
df.to_csv(csv_path, index=False)
print(f"Saved to {csv_path}")
