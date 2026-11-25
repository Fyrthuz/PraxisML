"""
Pipeline de preprocesamiento configurable para datasets tabulares.
Soporta imputación, escalado, encoding, feature engineering, descarte y
renombrado personalizado de variables, encadenando pasos por columna en orden
y generando nombres limpios sin prefijos técnicos de scikit-learn.
"""

import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
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
    "log_transform": lambda: FunctionTransformer(
        np.log1p, validate=True, feature_names_out="one-to-one"
    ),
    "polynomial": lambda: PolynomialFeatures(degree=2, include_bias=False),
    "binning": lambda target_bins=5: KBinsDiscretizer(
        n_bins=target_bins, encode="ordinal", strategy="quantile"
    ),
}


_CUSTOM_STEP_FACTORIES: Dict[str, Any] = {}


def register_preprocessing_step(step_type: str, factory: Any) -> None:
    """Registra un tipo de paso personalizado en el pipeline tabular."""
    _CUSTOM_STEP_FACTORIES[step_type.lower()] = factory


def _make_step_transformer(step: Dict[str, Any], index: int):
    step_type = step.get("type")

    if step_type and step_type.lower() in _CUSTOM_STEP_FACTORIES:
        return _CUSTOM_STEP_FACTORIES[step_type.lower()](step)

    if step_type == "impute":
        strategy = step.get("strategy") or step.get("method") or "mean"
        if strategy not in _IMPUTERS:
            raise ValueError(f"Paso {index}: estrategia de imputación no soportada: {strategy}")
        fill = step.get("fill_value", "MISSING")
        return _IMPUTERS[strategy]() if strategy != "constant" else _IMPUTERS[strategy](fill)

    if step_type == "scale":
        method = step.get("method") or step.get("strategy") or "standard"
        if method not in _SCALERS:
            raise ValueError(f"Paso {index}: método de escalado no soportado: {method}")
        return _SCALERS[method]()

    if step_type == "encode":
        method = step.get("method") or step.get("strategy") or "onehot"
        if method not in _ENCODERS:
            raise ValueError(f"Paso {index}: método de encoding no soportado: {method}")
        return _ENCODERS[method]()

    if step_type == "feature_eng":
        method = step.get("method") or step.get("strategy")
        if method not in _FEATURE_ENG:
            raise ValueError(f"Paso {index}: método feature_eng no soportado: {method}")
        if method == "binning":
            bins = step.get("bins", 5)
            return _FEATURE_ENG[method](target_bins=bins)
        return _FEATURE_ENG[method]()

    raise ValueError(f"Paso {index}: tipo de transformación desconocido: {step_type}")


def _clean_feature_names(raw_names: List[str]) -> List[str]:
    """Elimina prefijos técnicos de ColumnTransformer si existieran y evita colisiones."""
    cleaned: List[str] = []
    seen: Dict[str, int] = {}
    for name in raw_names:
        base = name.split("__", 1)[1] if "__" in name else name
        count = seen.get(base, 0)
        seen[base] = count + 1
        cleaned.append(base if count == 0 else f"{base}_{count + 1}")
    return cleaned


def _apply_column_renames(
    columns: List[str],
    rename_map: Dict[str, str],
) -> List[str]:
    if not rename_map:
        return columns

    normalized_map = {
        str(k).strip(): str(v).strip()
        for k, v in rename_map.items()
        if str(k).strip() and str(v).strip()
    }
    if not normalized_map:
        return columns

    renamed: List[str] = []
    for col in columns:
        new_name = normalized_map.get(col, col)
        renamed.append(new_name)

    if len(set(renamed)) != len(renamed):
        duplicates = sorted({c for c in renamed if renamed.count(c) > 1})
        raise ValueError(
            f"Los nombres personalizados generan columnas duplicadas: {', '.join(duplicates)}"
        )

    return renamed


def build_pipeline(config: Dict[str, Any], column_names: List[str]) -> ColumnTransformer:
    """
    Construye un ColumnTransformer a partir de la configuración JSON del usuario.

    Encadena automáticamente múltiples pasos sobre una misma columna en una
    sklearn Pipeline secuencial para evitar columnas duplicadas, usa nombres de
    salida limpios (`verbose_feature_names_out=False`) y adjunta el mapa de
    renombrado personalizado (`output_names` / `column_renames`).
    """
    steps = config.get("steps", [])
    if not steps:
        raise ValueError("La configuración debe tener al menos un paso en 'steps'.")

    merged_renames: Dict[str, str] = {}
    dropped_columns: List[str] = []
    per_column_steps: "OrderedDict[str, List[Tuple[str, Any]]]" = OrderedDict()
    multi_col_transformers: List[Tuple[str, Any, List[str]]] = []

    for i, step in enumerate(steps):
        step_type = step.get("type")
        columns = step.get("columns", [])

        if not columns:
            raise ValueError(f"Paso {i}: 'columns' es obligatorio y no puede estar vacío.")

        invalid = set(columns) - set(column_names)
        if invalid:
            raise ValueError(
                f"Paso {i}: columnas no encontradas en el dataset: {sorted(invalid)}"
            )

        step_output_names = step.get("output_names") or {}
        if isinstance(step_output_names, dict):
            for src, dst in step_output_names.items():
                if src and dst and str(dst).strip():
                    merged_renames[str(src).strip()] = str(dst).strip()

        if step_type == "drop":
            for col in columns:
                if col not in dropped_columns:
                    dropped_columns.append(col)
                per_column_steps.pop(col, None)
            continue

        if step_type == "rename":
            continue

        if step_type == "feature_eng" and (step.get("method") or step.get("strategy")) == "polynomial" and len(columns) > 1:
            multi_col_transformers.append(
                (f"feat_eng_{i}", _make_step_transformer(step, i), list(columns))
            )
            continue

        for col in columns:
            if col in dropped_columns:
                continue
            per_column_steps.setdefault(col, []).append(
                (f"{step_type}_{i}", _make_step_transformer(step, i))
            )

    top_renames = config.get("column_renames") or {}
    if isinstance(top_renames, dict):
        for src, dst in top_renames.items():
            if src and dst and str(dst).strip():
                merged_renames[str(src).strip()] = str(dst).strip()

    transformers: List[Tuple[str, Any, List[str]]] = []
    if dropped_columns:
        transformers.append(("drop_cols", "drop", dropped_columns))

    for col, col_steps in per_column_steps.items():
        if len(col_steps) == 1:
            step_name, transformer = col_steps[0]
            transformers.append((f"{step_name}_{col}", transformer, [col]))
        else:
            transformers.append((f"pipe_{col}", Pipeline(steps=col_steps), [col]))

    transformers.extend(multi_col_transformers)

    ct = ColumnTransformer(
        transformers=transformers,
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    ct._praxis_column_renames = merged_renames
    return ct


def apply_pipeline(
    pipeline: ColumnTransformer,
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    fit: bool = True,
    column_renames: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Ajusta y transforma (o solo transforma) el DataFrame con el pipeline,
    produciendo nombres limpios y aplicando cualquier renombrado personalizado.
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

    try:
        raw_out_columns = list(pipeline.get_feature_names_out())
        out_columns = _clean_feature_names(raw_out_columns)
    except Exception:
        out_columns = [f"feature_{i}" for i in range(transformed.shape[1])]

    effective_renames: Dict[str, str] = {}
    saved_renames = getattr(pipeline, "_praxis_column_renames", None)
    if isinstance(saved_renames, dict):
        effective_renames.update(saved_renames)
    if isinstance(column_renames, dict):
        effective_renames.update(column_renames)

    out_columns = _apply_column_renames(out_columns, effective_renames)

    df_out = pd.DataFrame(transformed, columns=out_columns, index=df_input.index)
    return df_out, y


def save_pipeline(
    pipeline: ColumnTransformer,
    pipeline_name: str,
    tenant_id: str,
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Serializa y guarda el pipeline en MLflow bajo el experimento de
    preprocesamiento del tenant actual, incluyendo los pasos y renombrados.
    """
    mlflow_svc = MLFlowService()
    mlflow_svc.ensure_tenant_experiment(tenant_id, "preprocessing")

    with mlflow.start_run(run_name=pipeline_name) as run:
        mlflow.sklearn.log_model(pipeline, "pipeline")

        if config:
            if "steps" in config:
                mlflow.set_tag("pipeline_steps", json.dumps(config["steps"]))
            if config.get("target_column"):
                mlflow.set_tag("target_column", str(config["target_column"]))
            if config.get("column_renames"):
                mlflow.set_tag("column_renames", json.dumps(config["column_renames"]))

        uri = f"runs:/{run.info.run_id}/pipeline"
        logger.info("Pipeline guardado en MLFlow: %s", uri)
        return uri


def load_pipeline(path_or_uri: str) -> ColumnTransformer:
    """Carga un pipeline desde MLflow o desde disco local."""
    if str(path_or_uri).startswith("runs:/"):
        mlflow_svc = MLFlowService()
        mlflow.set_tracking_uri(mlflow_svc.tracking_uri)
        return mlflow.sklearn.load_model(path_or_uri)

    if not Path(path_or_uri).exists():
        raise FileNotFoundError(f"Pipeline local no encontrado en disco: {path_or_uri}")
    return joblib.load(path_or_uri)
