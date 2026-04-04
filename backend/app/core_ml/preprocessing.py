"""
Pipeline de preprocesamiento configurable para datasets tabulares.
Soporta: escalado, imputación y encoding, con serialización vía joblib.
El preprocessor se guarda como artefacto MLFlow para reproducibilidad en inferencia.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    FunctionTransformer,
    KBinsDiscretizer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    RobustScaler,
    StandardScaler,
)

from app.services.mlflow_service import MLFlowService

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Registry de transformaciones disponibles
# ──────────────────────────────────────────────────────────────────────────────

_SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}

_IMPUTERS = {
    "mean": lambda: SimpleImputer(strategy="mean"),
    "median": lambda: SimpleImputer(strategy="median"),
    "most_frequent": lambda: SimpleImputer(strategy="most_frequent"),
    "constant": lambda fill="MISSING": SimpleImputer(strategy="constant", fill_value=fill),
}

_ENCODERS = {
    "onehot": lambda: OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    "ordinal": lambda: OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
}

_FEATURE_ENG = {
    "log_transform": lambda: FunctionTransformer(np.log1p, validate=True),
    "polynomial": lambda: PolynomialFeatures(degree=2, include_bias=False),
    "binning": lambda target_bins=5: KBinsDiscretizer(n_bins=target_bins, encode="ordinal", strategy="quantile"),
}



# ──────────────────────────────────────────────────────────────────────────────
# Configuración JSON esperada del usuario
# ──────────────────────────────────────────────────────────────────────────────
#
# Ejemplo de config:
# {
#   "target_column": "diagnosis",
#   "steps": [
#     {
#       "type": "impute",
#       "strategy": "mean",
#       "columns": ["age", "blood_pressure"]
#     },
#     {
#       "type": "scale",
#       "method": "standard",
#       "columns": ["age", "blood_pressure", "cholesterol"]
#     },
#     {
#       "type": "encode",
#       "method": "onehot",
#       "columns": ["gender", "smoking_status"]
#     }
#   ]
# }
# ──────────────────────────────────────────────────────────────────────────────


def build_pipeline(config: Dict[str, Any], column_names: List[str]) -> ColumnTransformer:
    """
    Construye un ColumnTransformer a partir de la configuración JSON del usuario.

    Args:
        config: Diccionario de configuración con claves 'steps' y opcionalmente 'target_column'.
        column_names: Lista de nombres de columnas del dataset original.

    Returns:
        sklearn ColumnTransformer listo para fit/transform.
    """
    steps = config.get("steps", [])
    if not steps:
        raise ValueError("La configuración debe tener al menos un paso en 'steps'.")

    transformers: List[Tuple[str, Any, List[str]]] = []

    for i, step in enumerate(steps):
        step_type = step.get("type")
        columns = step.get("columns", [])

        if not columns:
            raise ValueError(f"Paso {i}: 'columns' es obligatorio y no puede estar vacío.")

        # Validar que las columnas existen
        invalid = set(columns) - set(column_names)
        if invalid:
            raise ValueError(
                f"Paso {i}: columnas no encontradas en el dataset: {sorted(invalid)}"
            )

        if step_type == "drop":
            transformers.append((f"drop_{i}", "drop", columns))

        elif step_type == "impute":
            strategy = step.get("strategy", "mean")
            if strategy not in _IMPUTERS:
                raise ValueError(f"Paso {i}: estrategia de imputación no soportada: {strategy}")
            fill = step.get("fill_value", "MISSING")
            transformer = _IMPUTERS[strategy]() if strategy != "constant" else _IMPUTERS[strategy](fill)
            transformers.append((f"impute_{i}", transformer, columns))

        elif step_type == "scale":
            method = step.get("method", "standard")
            if method not in _SCALERS:
                raise ValueError(f"Paso {i}: método de escalado no soportado: {method}")
            transformers.append((f"scale_{i}", _SCALERS[method](), columns))

        elif step_type == "encode":
            method = step.get("method", "onehot")
            if method not in _ENCODERS:
                raise ValueError(f"Paso {i}: método de encoding no soportado: {method}")
            transformers.append((f"encode_{i}", _ENCODERS[method](), columns))

        elif step_type == "feature_eng":
            method = step.get("method")
            if method not in _FEATURE_ENG:
                raise ValueError(f"Paso {i}: método feature_eng no soportado: {method}")
            if method == "binning":
                bins = step.get("bins", 5)
                transformer = _FEATURE_ENG[method](target_bins=bins)
            else:
                transformer = _FEATURE_ENG[method]()
            transformers.append((f"feat_eng_{i}", transformer, columns))

        else:
            raise ValueError(f"Paso {i}: tipo de transformación desconocido: {step_type}")

    # remainder='passthrough' mantiene las columnas no transformadas
    ct = ColumnTransformer(transformers=transformers, remainder="passthrough")
    return ct


def apply_pipeline(
    pipeline: ColumnTransformer,
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    fit: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Ajusta y transforma (o solo transforma) el DataFrame con el pipeline.

    Args:
        pipeline: ColumnTransformer construido por build_pipeline.
        df: DataFrame original.
        target_column: Nombre de la columna objetivo (se separa antes de transformar).
        fit: Si es True, hace fit_transform. Si es False, solo transform (inferencia).

    Returns:
        (df_transformed, y) donde y es la serie objetivo si se especificó.
    """
    y = None
    df_input = df.copy()

    if target_column:
        if target_column not in df_input.columns:
            raise ValueError(f"Columna objetivo '{target_column}' no encontrada en el dataset.")
        y = df_input.pop(target_column)

    if fit:
        transformed = pipeline.fit_transform(df_input)
    else:
        transformed = pipeline.transform(df_input)

    # Obtener nombres de columnas del output
    try:
        out_columns = pipeline.get_feature_names_out()
    except Exception:
        out_columns = [f"feature_{i}" for i in range(transformed.shape[1])]

    df_out = pd.DataFrame(transformed, columns=out_columns, index=df_input.index)
    return df_out, y


def save_pipeline(pipeline: ColumnTransformer, pipeline_name: str, tenant_id: str, config: Optional[Dict[str, Any]] = None) -> str:
    """
    Serializa y guarda el pipeline como un modelo en MLFlow bajo
    el experimento de preprocesamiento del tenant actual.

    Returns:
        A MLFlow models URI, por ejemplo: `runs:/<run_id>/pipeline`
    """
    mlflow_svc = MLFlowService()
    mlflow.set_tracking_uri(mlflow_svc.tracking_uri)

    experiment_name = f"tenant_{tenant_id}_preprocessing"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=pipeline_name) as run:
        mlflow.sklearn.log_model(pipeline, "pipeline")

        # Log config steps as tags for easier visibility in the UI and retrieval
        if config:
            if "steps" in config:
                mlflow.set_tag("pipeline_steps", json.dumps(config["steps"]))
            if "target_column" in config:
                mlflow.set_tag("target_column", str(config["target_column"]))

        uri = f"runs:/{run.info.run_id}/pipeline"
        logger.info("Pipeline guardado en MLFlow: %s", uri)
        return uri


def load_pipeline(path_or_uri: str) -> ColumnTransformer:
    """
    Carga un pipeline desde MLFlow o desde el disco (compatibilidad Legacy).
    """
    if str(path_or_uri).startswith("runs:/"):
        # Localizamos MLFlow
        mlflow_svc = MLFlowService()
        mlflow.set_tracking_uri(mlflow_svc.tracking_uri)
        return mlflow.sklearn.load_model(path_or_uri)
    else:
        # Legacy fallback para ficheros locales
        if not Path(path_or_uri).exists():
            raise FileNotFoundError(f"Pipeline local no encontrado en disco: {path_or_uri}")
        return joblib.load(path_or_uri)
