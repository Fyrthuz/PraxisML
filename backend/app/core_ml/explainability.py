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
    background: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    task_type: str = "classification",
) -> Dict[str, Any]:
    """
    Calcula SHAP values usando KernelExplainer para cualquier modelo.

    Args:
        model: Modelo scikit-learn o PyTorch (con método predict)
        data: Datos de entrada (numpy array o DataFrame)
        feature_names: Nombres de las características
        background_samples: Número de muestras para background dataset
        background: Dataset de referencia (opcional). Si no se provee, se usa una muestra de 'data'.
        task_type: Tipo de tarea ('classification' o 'regression')

    Returns:
        Dict con 'shap_values' (lista de arrays) y 'expected_value'
    """
    try:
        import torch

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

        # Definir una función de predicción robusta que maneje la conversión NumPy -> Tensor
        # y devuelva probabilidades/logits continuos para mejores explicaciones.
        def predict_function(x):
            try:
                # 1. Caso PyTorch (nn.Module o TorchScript)
                if hasattr(model, "forward") or isinstance(model, (torch.nn.Module, torch.jit.ScriptModule, torch.jit.RecursiveScriptModule)):
                    device = next(model.parameters()).device if hasattr(model, "parameters") and list(model.parameters()) else torch.device("cpu")
                    # Convertir input (NumPy de SHAP) a Tensor
                    tensor_x = torch.from_numpy(x).float().to(device)
                    
                    model.eval()
                    with torch.no_grad():
                        output = model(tensor_x)
                        
                        # Si la salida es 1D (regresión o binary proba única), asegurar 2D
                        if output.dim() == 1:
                            output = output.unsqueeze(1)
                        
                        # Si es clasificación con una sola salida (probabilidad binaria)
                        if task_type == "classification" and output.shape[1] == 1:
                            # SHAP prefiere ver las dos clases [p0, p1] para evitar gradientes planos
                            p1 = output
                            p0 = 1.0 - p1
                            output = torch.cat([p0, p1], dim=1)
                        
                        return output.cpu().numpy()
                
                # 2. Caso MLflow model wrapper o Sklearn
                if task_type == "classification" and hasattr(model, "predict_proba"):
                    return model.predict_proba(x)
                
                if hasattr(model, "predict"):
                    return model.predict(x)
                
                raise ValueError("El modelo no tiene un método de predicción identificable.")
            except Exception as e:
                logger.error(f"Error en predict_function durante SHAP: {e}")
                raise

        # Asegurar formatos numpy para KernelExplainer
        data_np = np.asarray(data).astype(np.float32)

        # Determinar background dataset para KernelExplainer
        if background is not None:
            if isinstance(background, pd.DataFrame):
                # Convertir categóricas a numéricas por si acaso
                for col in background.select_dtypes(['object', 'category']).columns:
                    background[col] = background[col].astype('category').cat.codes
                background_np = background.to_numpy().astype(np.float32)
            else:
                background_np = np.asarray(background).astype(np.float32)
            
            # Limitar tamaño del background si es muy grande
            if len(background_np) > background_samples:
                background_np = shap.sample(background_np, background_samples)
            logger.info(f"SHAP: Usando background externo de {len(background_np)} muestras.")
        else:
            # Si no hay background externo, usar muestra de 'data'
            if len(data_np) > background_samples:
                background_np = shap.sample(data_np, background_samples)
            else:
                background_np = data_np
            logger.info(f"SHAP: Usando 'data' como su propio background ({len(background_np)} muestras).")

        # Validación final de background para evitar ceros
        if len(background_np) <= 1:
            if len(data_np) == 1 and np.array_equal(background_np, data_np):
                logger.warning("SHAP: TRAP DETECTED. El background es idéntico al dato a explicar. Los valores SHAP serán 0.")
            else:
                logger.warning(f"SHAP: Background muy pequeño ({len(background_np)} muestras). La explicación puede no ser fiable.")

        # Crear explainer con la función robusta
        explainer = shap.KernelExplainer(predict_function, background_np)

        # Calcular SHAP values
        shap_values = explainer.shap_values(data_np)

        # Convertir a lista si es numpy array
        if isinstance(shap_values, np.ndarray):
            shap_values_list = shap_values.tolist()
        elif isinstance(shap_values, list):
            shap_values_list = [sv.tolist() if isinstance(sv, np.ndarray) else sv for sv in shap_values]
        else:
            shap_values_list = shap_values

        return {
            "shap_values": shap_values_list,
            "expected_value": explainer.expected_value.tolist() if hasattr(explainer.expected_value, "tolist") else explainer.expected_value,
            "feature_names": feature_names,
        }
    except Exception as e:
        logger.error(f"Error calculando SHAP values: {e}", exc_info=True)
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
