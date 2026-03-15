"""
Módulo de explicabilidad para modelos tabulares usando SHAP.
Soporta KernelExplainer para todos los modelos (árboles, lineales, SVM, KNN, PyTorch MLP).
"""

import shap
import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def get_shap_values(
    model: Any,
    data: Union[np.ndarray, pd.DataFrame],
    feature_names: Optional[List[str]] = None,
    background_samples: int = 100,
) -> Dict[str, Any]:
    """
    Calcula SHAP values usando KernelExplainer para cualquier modelo.

    Args:
        model: Modelo scikit-learn o PyTorch (con método predict)
        data: Datos de entrada (numpy array o DataFrame)
        feature_names: Nombres de las características
        background_samples: Número de muestras para background dataset

    Returns:
        Dict con 'shap_values' (lista de arrays) y 'expected_value'
    """
    try:
        # Convertir a numpy si es DataFrame
        if isinstance(data, pd.DataFrame):
            if feature_names is None:
                feature_names = data.columns.tolist()
            data_np = data.values
        else:
            data_np = data

        # Si feature_names no se proporciona, generar nombres genéricos
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(data_np.shape[1])]

        # Crear wrapper para modelo si es necesario
        if hasattr(model, "predict"):
            # Modelo scikit-learn
            predictor = model.predict
        else:
            # Modelo PyTorch desde MLflow (debe tener método predict)
            if not hasattr(model, "predict"):
                raise ValueError(
                    "Modelo no tiene método predict. Se requiere método predict para SHAP."
                )
            predictor = model.predict

        # Crear background dataset para KernelExplainer
        if len(data_np) > background_samples:
            background = shap.sample(data_np, background_samples)
        else:
            background = data_np

        # Crear explainer
        explainer = shap.KernelExplainer(predictor, background)

        # Calcular SHAP values
        shap_values = explainer.shap_values(data_np)

        # Convertir a lista si es numpy array
        if isinstance(shap_values, np.ndarray):
            shap_values_list = shap_values.tolist()
        else:
            shap_values_list = [sv.tolist() for sv in shap_values]

        return {
            "shap_values": shap_values_list,
            "expected_value": explainer.expected_value,
            "feature_names": feature_names,
        }
    except Exception as e:
        logger.error(f"Error calculando SHAP values: {e}")
        raise


def explain_prediction(
    model: Any, row_data: Dict[str, Any], feature_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Explica una predicción individual usando SHAP.

    Args:
        model: Modelo entrenado
        row_data: Diccionario con valores de características
        feature_names: Nombres de características (opcional)

    Returns:
        Dict con explicación de la predicción
    """
    try:
        # Convertir a DataFrame
        df = pd.DataFrame([row_data])

        # Obtener feature names
        if feature_names is None:
            feature_names = df.columns.tolist()

        # Calcular SHAP values
        result = get_shap_values(model, df, feature_names)

        # Calcular importancia relativa (suma absoluta de SHAP values)
        shap_values = np.array(result["shap_values"][0])
        importance = np.abs(shap_values).sum()

        return {
            "shap_values": result["shap_values"][0],
            "expected_value": result["expected_value"],
            "feature_names": feature_names,
            "total_importance": float(importance),
            "positive_contributors": [
                {"feature": feature_names[i], "value": shap_values[i]}
                for i in range(len(shap_values))
                if shap_values[i] > 0
            ],
            "negative_contributors": [
                {"feature": feature_names[i], "value": shap_values[i]}
                for i in range(len(shap_values))
                if shap_values[i] < 0
            ],
        }
    except Exception as e:
        logger.error(f"Error explicando predicción: {e}")
        raise
