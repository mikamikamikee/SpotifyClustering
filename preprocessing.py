import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from pathlib import Path
import joblib

# -------------------------------
# 1. Load dataset
# -------------------------------
data_path = Path("SpotifyFeatures.csv")  # adjust path if needed
df = pd.read_csv(data_path)

print("Dataset shape (rows, columns):", df.shape)
print("Columns:\n", df.columns.tolist())
print("\nData types:")
print(df.dtypes)

# -------------------------------
# 2. Define feature + metadata columns
# -------------------------------

# Audio features to use for clustering
audio_feature_cols = [
    "danceability",
    "energy",
    "valence",
    "tempo",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "key",   # will be circular-encoded
    "mode",  # will be encoded to 0/1
]

# Sanity check: make sure these columns exist
missing_cols = [c for c in audio_feature_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"These expected feature columns are missing from the dataset: {missing_cols}")

# Optional: metadata to keep for later interpretation (not used in clustering)
metadata_cols = [
    "id",
    "track_name",
    "artist_name"
]
metadata_cols = [c for c in metadata_cols if c in df.columns]

# Subset
df_features = df[audio_feature_cols].copy()
df_meta = df[metadata_cols].copy() if metadata_cols else None

print("\nFeature dataframe shape:", df_features.shape)
if df_meta is not None:
    print("Metadata dataframe shape:", df_meta.shape)

# -------------------------------
# 3. Encode 'mode' (Major/Minor) to numeric
# -------------------------------

print("\nOriginal 'mode' dtype:", df_features["mode"].dtype)
print("Sample 'mode' values:")
print(df_features["mode"].head(10))

# Map possible string values to 0/1
mode_map = {
    "Minor": 0,
    "Major": 1
}

df_features["mode"] = df_features["mode"].replace(mode_map)

# If there are still non-numeric values (e.g., weird labels), try to coerce
df_features["mode"] = pd.to_numeric(df_features["mode"], errors="coerce")

# Drop rows where mode is still NaN after mapping (invalid / unknown)
before_mode_drop = df_features.shape[0]
valid_mode_mask = df_features["mode"].notna()
df_features = df_features[valid_mode_mask].copy()
if df_meta is not None:
    df_meta = df_meta.loc[df_features.index].copy()
after_mode_drop = df_features.shape[0]

print(f"\nDropped {before_mode_drop - after_mode_drop} rows with invalid 'mode' values.")
print("Mode unique values after encoding:", df_features["mode"].unique())

# Ensure integer type
df_features["mode"] = df_features["mode"].astype(int)

# -------------------------------
# 4. Circular encoding of key
# -------------------------------

# Mapping from note names to integers (if your 'key' column uses note names)
key_map = {
    "C": 0,
    "C#": 1,
    "D": 2,
    "D#": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "G": 7,
    "G#": 8,
    "A": 9,
    "A#": 10,
    "B": 11
}

df_enc = df_features.copy()

print("\nOriginal 'key' dtype:", df_enc["key"].dtype)
print("Sample 'key' values:")
print(df_enc["key"].head(10))

# Case A: 'key' is already numeric (e.g., 0-11)
if np.issubdtype(df_enc["key"].dtype, np.number):
    df_enc["key"] = df_enc["key"].astype(float).astype("Int64")

# Case B: 'key' is non-numeric (e.g., "C", "D#", etc.) → map using key_map
else:
    df_enc["key"] = df_enc["key"].astype(str).str.strip()
    df_enc["key"] = df_enc["key"].map(key_map).astype("Int64")

print("\nAfter cleaning/mapping, sample 'key' values:")
print(df_enc["key"].head(10))

# Drop rows where key is still missing/invalid
before_key_drop = df_enc.shape[0]
valid_key_mask = df_enc["key"].notna()
df_enc = df_enc[valid_key_mask].copy()
if df_meta is not None:
    df_meta = df_meta.loc[df_enc.index].copy()
after_key_drop = df_enc.shape[0]

print(f"\nDropped {before_key_drop - after_key_drop} rows with missing/invalid key values.")
print("Shape after dropping invalid keys:", df_enc.shape)

# Convert 'key' to plain int
df_enc["key"] = df_enc["key"].astype(int)

# Compute angle in radians (12 semitones on a circle)
key_angle = 2 * np.pi * (df_enc["key"] / 12.0)

# Circular encoding
df_enc["key_sin"] = np.sin(key_angle)
df_enc["key_cos"] = np.cos(key_angle)

# Drop original 'key' column
df_enc = df_enc.drop(columns=["key"])

print("\nColumns after circular key encoding:")
print(df_enc.columns.tolist())
print("\nFirst 5 rows of key_sin/key_cos:")
print(df_enc[["key_sin", "key_cos"]].head())

# -------------------------------
# 5. Optional outlier clipping
# -------------------------------

df_clean = df_enc.copy()

def clip_outliers(series, lower_quantile=0.01, upper_quantile=0.99):
    """Clip values of a pandas Series to the given quantile range."""
    lower = series.quantile(lower_quantile)
    upper = series.quantile(upper_quantile)
    return series.clip(lower=lower, upper=upper)

# Features where outliers can be strong
features_to_clip = ["tempo", "loudness", "speechiness", "instrumentalness"]

for col in features_to_clip:
    if col in df_clean.columns:
        df_clean[col] = clip_outliers(df_clean[col])

print("\nSummary stats after optional clipping:")
print(df_clean[features_to_clip].describe().T)

# -------------------------------
# 6. Standard scaling
# -------------------------------

# Final feature columns after key encoding and clipping
final_feature_cols = df_clean.columns.tolist()
print("\nFinal feature columns used for clustering:")
print(final_feature_cols)

# Convert to numpy array
X = df_clean.values

# Build scaling pipeline
preprocess_pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
    ]
)

# Fit and transform
X_preprocessed = preprocess_pipeline.fit_transform(X)

print("\nPreprocessed matrix shape:", X_preprocessed.shape)
print("First 5 feature means (approx 0):", X_preprocessed.mean(axis=0)[:5])
print("First 5 feature stds (approx 1):", X_preprocessed.std(axis=0)[:5])

# -------------------------------
# 7. Wrap back into DataFrame and save
# -------------------------------

df_preprocessed = pd.DataFrame(
    X_preprocessed,
    columns=final_feature_cols,
    index=df_clean.index  # keep original index so it matches metadata
)

print("\nPreview of preprocessed features:")
print(df_preprocessed.head())

# Reset index for clean saving & alignment
df_preprocessed = df_preprocessed.reset_index(drop=True)
if df_meta is not None:
    df_meta = df_meta.reset_index(drop=True)

# Save preprocessed features and metadata
df_preprocessed.to_csv("spotify_preprocessed_features.csv", index=False)
print("\nSaved preprocessed features to 'spotify_preprocessed_features.csv'.")

if df_meta is not None:
    df_meta.to_csv("spotify_metadata_aligned.csv", index=False)
    print("Saved aligned metadata to 'spotify_metadata_aligned.csv'.")

# Save the scaler pipeline
joblib.dump(preprocess_pipeline, "preprocess_scaler_pipeline.joblib")
print("Saved scaler pipeline to 'preprocess_scaler_pipeline.joblib'.")
