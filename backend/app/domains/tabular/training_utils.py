"""
Utilidades para preprocesamiento de features en entrenamiento.
"""

import pandas as pd


def prepare_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte features no numéricas a numéricas (encoding simple).
    Completa valores NaN con la mediana.
    """
    X_clean = X.copy()
    for col in X_clean.select_dtypes(include=["object", "category"]).columns:
        X_clean[col] = X_clean[col].astype("category").cat.codes
    for col in X_clean.select_dtypes(include=["number"]).columns:
        if X_clean[col].isna().any():
            X_clean[col] = X_clean[col].fillna(X_clean[col].median())
    return X_clean
