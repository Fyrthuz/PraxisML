"""
Módulo de explicabilidad para modelos tabulares usando SHAP.
Soporta KernelExplainer para todos los modelos (árboles, lineales, SVM, KNN, PyTorch MLP).
"""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import shap

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


def _extract_1d_shap(raw_sv: Any, row_idx: int = 0) -> List[float]:
    """Normaliza la salida de SHAP (1D, 2D o 3D multiclase) a un vector 1D por fila."""
    if isinstance(raw_sv, list):
        # Lista por clase (clasificación binaria/multiclase): usar clase positiva (índice 1 si existe)
        chosen = raw_sv[1] if len(raw_sv) > 1 else raw_sv[0]
        arr = np.asarray(chosen)
    else:
        arr = np.asarray(raw_sv)

    if arr.ndim == 3:
        # (n_samples, n_features, n_classes)
        class_idx = 1 if arr.shape[2] > 1 else 0
        vec = arr[row_idx, :, class_idx]
    elif arr.ndim == 2:
        vec = arr[row_idx]
    else:
        vec = arr
    return [round(float(v), 6) for v in vec]


def _extract_expected_scalar(expected: Any) -> float:
    if expected is None:
        return 0.0
    arr = np.asarray(expected).ravel()
    if arr.size == 0:
        return 0.0
    return round(float(arr[1] if arr.size > 1 else arr[0]), 6)


def explain_tabular_fast(
    model: Any,
    data: Union[np.ndarray, pd.DataFrame],
    feature_names: Optional[List[str]] = None,
    task_type: str = "classification",
    max_rows: int = 5,
) -> List[Dict[str, Any]]:
    """
    Explicabilidad tabular rápida y compatible con inferencia online (1 o más filas).

    Prioridad:
      1. `shap.TreeExplainer` para modelos basados en árboles (exacto, ~3 ms incluso con 1 fila).
      2. Atribución lineal exacta (`coef_ * x`) para modelos lineales.
      3. Perturbación por feature (`f(x) - f(x_i=0)`) como fallback universal sin background trap.
    """
    if isinstance(data, pd.DataFrame):
        if feature_names is None:
            feature_names = [str(c) for c in data.columns.tolist()]
        X_df = data.iloc[:max_rows].copy()
        for col in X_df.select_dtypes(["object", "category"]).columns:
            X_df[col] = X_df[col].astype("category").cat.codes
        X_np = X_df.to_numpy(dtype=np.float32)
    else:
        X_np = np.asarray(data, dtype=np.float32)
        if X_np.ndim == 1:
            X_np = X_np.reshape(1, -1)
        X_np = X_np[:max_rows]
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X_np.shape[1])]

    n_rows, n_feats = X_np.shape
    names = list(feature_names)[:n_feats] + [
        f"feature_{i}" for i in range(len(feature_names), n_feats)
    ]

    # 1. TreeExplainer (RandomForest, GradientBoosting, DecisionTree, XGBoost, LightGBM...)
    try:
        is_tree = any(
            hasattr(model, attr)
            for attr in ("tree_", "estimators_", "get_booster", "booster_")
        )
        if is_tree:
            explainer = shap.TreeExplainer(model)
            raw_sv = explainer.shap_values(X_np)
            expected = _extract_expected_scalar(explainer.expected_value)
            out: List[Dict[str, Any]] = []
            for r in range(n_rows):
                contribs = _extract_1d_shap(raw_sv, r)
                feats = sorted(
                    [
                        {
                            "name": names[i],
                            "value": round(float(X_np[r, i]), 6),
                            "contribution": contribs[i],
                        }
                        for i in range(n_feats)
                    ],
                    key=lambda item: abs(item["contribution"]),
                    reverse=True,
                )
                out.append(
                    {
                        "method": "shap_tree",
                        "expected_value": expected,
                        "shap_values": contribs,
                        "feature_names": names,
                        "features": feats,
                    }
                )
            return out
    except Exception as exc:
        logger.debug("TreeExplainer fallback (%s)", exc)

    # 2. Modelos lineales con coef_
    if hasattr(model, "coef_"):
        try:
            coef = np.asarray(model.coef_, dtype=np.float64)
            if coef.ndim > 1:
                coef = coef[0]
            intercept = _extract_expected_scalar(getattr(model, "intercept_", 0.0))
            out = []
            for r in range(n_rows):
                contribs = [round(float(coef[i] * X_np[r, i]), 6) for i in range(n_feats)]
                feats = sorted(
                    [
                        {
                            "name": names[i],
                            "value": round(float(X_np[r, i]), 6),
                            "contribution": contribs[i],
                        }
                        for i in range(n_feats)
                    ],
                    key=lambda item: abs(item["contribution"]),
                    reverse=True,
                )
                out.append(
                    {
                        "method": "linear_attribution",
                        "expected_value": intercept,
                        "shap_values": contribs,
                        "feature_names": names,
                        "features": feats,
                    }
                )
            return out
        except Exception as exc:
            logger.debug("Linear attribution fallback (%s)", exc)

    # 3. Fallback universal: perturbación por oclusión (f(x) - f(x con feature=0))
    def _score_row(arr_2d: np.ndarray) -> float:
        try:
            import torch

            if isinstance(model, torch.nn.Module):
                with torch.no_grad():
                    out_t = model(torch.from_numpy(arr_2d).float())
                    if isinstance(out_t, (tuple, list)):
                        out_t = out_t[0]
                    vals = out_t.detach().cpu().numpy().ravel()
                    return float(vals[1] if vals.size > 1 else vals[0])
        except Exception:
            pass
        if task_type == "classification" and hasattr(model, "predict_proba"):
            probs = np.asarray(model.predict_proba(arr_2d)).ravel()
            return float(probs[1] if probs.size > 1 else probs[0])
        pred = np.asarray(model.predict(arr_2d)).ravel()
        return float(pred[0])

    out = []
    zero_baseline = _score_row(np.zeros((1, n_feats), dtype=np.float32))
    for r in range(n_rows):
        row = X_np[r : r + 1]
        base_score = _score_row(row)
        contribs = []
        for i in range(n_feats):
            perturbed = row.copy()
            perturbed[0, i] = 0.0
            delta = base_score - _score_row(perturbed)
            contribs.append(round(float(delta), 6))
        feats = sorted(
            [
                {
                    "name": names[i],
                    "value": round(float(X_np[r, i]), 6),
                    "contribution": contribs[i],
                }
                for i in range(n_feats)
            ],
            key=lambda item: abs(item["contribution"]),
            reverse=True,
        )
        out.append(
            {
                "method": "feature_perturbation",
                "expected_value": round(zero_baseline, 6),
                "shap_values": contribs,
                "feature_names": names,
                "features": feats,
            }
        )
    return out


def explain_nlp_text(
    service: Any,
    handle: Dict[str, Any],
    text: str,
    base_result: Dict[str, Any],
    max_words: int = 16,
) -> Dict[str, Any]:
    """
    Explicabilidad NLP por oclusión de palabras (leave-one-out).
    Mide cuánto cae la confianza de la clase predicha al omitir cada palabra.
    """
    words = [w for w in str(text).split() if w.strip()]
    base_label = base_result.get("label")
    base_conf = float(base_result.get("confidence") or 0.0)
    if not words or base_label is None:
        return {
            "method": "token_occlusion",
            "base_label": base_label,
            "base_confidence": round(base_conf, 6),
            "tokens": [],
        }

    step = max(1, len(words) // max_words) if len(words) > max_words else 1
    eval_indices = list(range(0, len(words), step))[:max_words]
    token_scores: List[Dict[str, Any]] = []

    for idx in eval_indices:
        reduced = " ".join(w for j, w in enumerate(words) if j != idx).strip()
        if not reduced:
            contrib = base_conf
        else:
            res_occ = service.predict(handle, reduced)
            occ_conf = float(res_occ.get("confidence") or 0.0)
            if res_occ.get("label") == base_label:
                contrib = base_conf - occ_conf
            else:
                # Si quitar la palabra invierte la predicción, su contribución es fuertemente positiva
                contrib = base_conf + occ_conf * 0.5
        token_scores.append(
            {
                "token": words[idx],
                "position": idx,
                "contribution": round(float(contrib), 6),
            }
        )

    ranked = sorted(token_scores, key=lambda t: abs(t["contribution"]), reverse=True)
    return {
        "method": "token_occlusion",
        "base_label": base_label,
        "base_confidence": round(base_conf, 6),
        "tokens": token_scores,
        "top_tokens": ranked[:8],
    }
