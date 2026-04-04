from typing import Any, Dict

import numpy as np
import pandas as pd


def detect_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """
    Detects the semantic type of each column in a DataFrame.
    Returns a dict mapping column name -> 'numeric', 'categorical', or 'datetime'
    """
    types = {}
    for col in df.columns:
        dtype = df[col].dtype

        if pd.api.types.is_datetime64_any_dtype(dtype):
            types[col] = "datetime"
        elif pd.api.types.is_numeric_dtype(dtype):
            # If a numeric column has very few unique values, it might be categorical encoded as numeric (e.g., 0/1)
            # We'll treat it as numeric for profiling, but the user can encode it later.
            types[col] = "numeric"
        else:
            types[col] = "categorical"

    return types

def profile_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generates a statistical profile for each column in the DataFrame.
    """
    col_types = detect_column_types(df)
    profile = {}

    total_rows = len(df)

    for col in df.columns:
        col_type = col_types[col]
        series = df[col]
        null_count = int(series.isna().sum())
        null_pct = round((null_count / total_rows) * 100, 2) if total_rows > 0 else 0

        col_stats = {
            "type": col_type,
            "null_count": null_count,
            "null_pct": null_pct,
            "distinct_count": series.nunique(dropna=True),
        }

        if col_type == "numeric":
            valid_data = series.dropna()
            if not valid_data.empty:
                col_stats.update({
                    "min": float(valid_data.min()),
                    "max": float(valid_data.max()),
                    "mean": float(valid_data.mean()),
                    "median": float(valid_data.median()),
                    "std": float(valid_data.std()) if len(valid_data) > 1 else 0.0,
                    "zeros": int((valid_data == 0).sum()),
                })
                # simple 10-bin histogram
                hist, bin_edges = np.histogram(valid_data, bins=10)
                col_stats["histogram"] = {
                    "counts": hist.tolist(),
                    "bins": bin_edges.tolist()
                }

        elif col_type == "categorical":
            valid_data = series.dropna()
            if not valid_data.empty:
                val_counts = valid_data.value_counts().head(10)
                col_stats["top_values"] = [
                    {"value": str(k), "count": int(v)}
                    for k, v in val_counts.items()
                ]

        elif col_type == "datetime":
            valid_data = series.dropna()
            if not valid_data.empty:
                col_stats.update({
                    "min": str(valid_data.min()),
                    "max": str(valid_data.max()),
                })

        profile[col] = col_stats

    return profile
