"""
ModelRuntime & ModelRuntimeRegistry — Capa de serving agnóstica al framework.

Desacopla la ejecución de inferencia en Model Endpoints de librerías concretas:
  - `sklearn`: modelos scikit-learn / XGBoost / LightGBM / CatBoost con API `.predict`
  - `pytorch`: módulos PyTorch / TorchScript
  - `nlp`: modelos Transformers / ONNX NLP
  - `pyfunc`: runtime universal MLflow pyfunc para cualquier librería personalizada
Extensible mediante `register_model_runtime(name, runtime)` (e.g. ONNX, Triton, vLLM).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from app.models.ml_model import MLModel
from app.domains.tabular.inference_service import get_inference_service
from app.platform.cache.model_cache import get_model_cache

logger = logging.getLogger(__name__)


def _load_cached_pipeline(pipeline_path: str):
    cache = get_model_cache()
    cache_key = f"pipeline:{pipeline_path}"
    pipe = cache.get(cache_key)
    if pipe is None:
        from app.core_ml.preprocessing import load_pipeline

        pipe = load_pipeline(pipeline_path)
        cache.set(cache_key, pipe, ttl=600)
    return pipe


def prepare_tabular_dataframe(
    rows: List[Dict[str, Any]],
    ml_model: MLModel,
    pipeline_path: Optional[str],
) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    meta = ml_model.metrics_metadata or {}
    expected_features: List[str] = list(meta.get("feature_names") or [])

    if pipeline_path:
        already_preprocessed = bool(expected_features) and all(
            col in df.columns for col in expected_features
        )
        if not already_preprocessed:
            pipe = _load_cached_pipeline(pipeline_path)
            if hasattr(pipe, "feature_names_in_"):
                for col in pipe.feature_names_in_:
                    if col not in df.columns:
                        df[col] = np.nan
                df = df[list(pipe.feature_names_in_)]
            from app.core_ml.preprocessing import apply_pipeline

            df, _ = apply_pipeline(pipe, df, target_column=None, fit=False)

    if expected_features:
        for col in expected_features:
            if col not in df.columns:
                df[col] = 0.0
        df = df[expected_features]

    return df


class ModelRuntime(ABC):
    """Contrato común para ejecutar inferencia sobre un tipo/flavor de modelo."""

    runtime_name: str

    @abstractmethod
    def can_handle(self, ml_model: MLModel) -> bool:
        ...

    @abstractmethod
    def predict(
        self,
        ml_model: MLModel,
        inputs: Union[Dict[str, Any], List[Dict[str, Any]], List[str]],
        pipeline_path: Optional[str] = None,
        explain: bool = False,
    ) -> Dict[str, Any]:
        """Devuelve {'predictions': [...], 'probabilities': [...] | None, 'count': int, 'explanations'?: [...]}."""
        ...


class NLPRuntime(ModelRuntime):
    runtime_name = "nlp"

    def can_handle(self, ml_model: MLModel) -> bool:
        meta = ml_model.metrics_metadata or {}
        return meta.get("domain") == "nlp" or meta.get("framework") in {
            "huggingface",
            "onnx_nlp",
        }

    def predict(
        self,
        ml_model: MLModel,
        inputs: Union[Dict[str, Any], List[Dict[str, Any]], List[str]],
        pipeline_path: Optional[str] = None,
        explain: bool = False,
    ) -> Dict[str, Any]:
        from app.domains.nlp.inference import get_nlp_inference_service

        if isinstance(inputs, dict):
            texts = [str(inputs.get("text") or inputs.get("input") or "")]
        elif isinstance(inputs, list):
            texts = [
                str(item.get("text") or item.get("input") or "")
                if isinstance(item, dict)
                else str(item)
                for item in inputs
            ]
        else:
            texts = [str(inputs)]

        svc = get_nlp_inference_service()
        handle = svc.load_model(ml_model)
        results = [svc.predict(handle, text) for text in texts]
        probabilities = [r.get("confidence") for r in results if r.get("confidence") is not None]
        explanations = None
        if explain:
            try:
                from app.core_ml.explainability import explain_nlp_text

                explanations = [
                    explain_nlp_text(svc, handle, text, res)
                    for text, res in zip(texts[:5], results[:5])
                ]
            except Exception as exc:
                logger.warning("NLP serving explainability skipped: %s", exc)
        return {
            "predictions": results,
            "probabilities": probabilities or None,
            "count": len(results),
            "explanations": explanations,
        }


class TabularMLRuntime(ModelRuntime):
    """Runtime tabular compatible con scikit-learn, XGBoost, LightGBM, PyTorch y pyfunc."""

    runtime_name = "tabular_generic"

    def can_handle(self, ml_model: MLModel) -> bool:
        return True

    def predict(
        self,
        ml_model: MLModel,
        inputs: Union[Dict[str, Any], List[Dict[str, Any]], List[str]],
        pipeline_path: Optional[str] = None,
        explain: bool = False,
    ) -> Dict[str, Any]:
        rows = [inputs] if isinstance(inputs, dict) else list(inputs)
        if not rows or not all(isinstance(r, dict) for r in rows):
            raise ValueError(
                "Los modelos tabulares requieren `inputs` como objeto JSON o lista de objetos JSON."
            )

        inference_svc = get_inference_service()
        loaded_model = inference_svc.load_model(ml_model)
        df = prepare_tabular_dataframe(rows, ml_model, pipeline_path)
        task_type = (ml_model.metrics_metadata or {}).get("task_type") or "classification"

        def _maybe_explain() -> Optional[List[Dict[str, Any]]]:
            if not explain:
                return None
            try:
                from app.core_ml.explainability import explain_tabular_fast

                return explain_tabular_fast(
                    loaded_model,
                    df,
                    feature_names=list(df.columns),
                    task_type=task_type,
                )
            except Exception as exc:
                logger.warning("Tabular serving explainability skipped: %s", exc)
                return None

        # PyTorch nn.Module / TorchScript
        try:
            import torch

            if isinstance(loaded_model, torch.nn.Module):
                X_np = df.to_numpy().astype(np.float32)
                tensor = torch.tensor(X_np, dtype=torch.float32)
                with torch.no_grad():
                    out = loaded_model(tensor)
                    if isinstance(out, (tuple, list)):
                        out = out[0]
                    out_np = out.detach().cpu().numpy()

                probabilities = None
                if out_np.ndim == 2 and out_np.shape[1] > 1:
                    exps = np.exp(out_np - np.max(out_np, axis=1, keepdims=True))
                    probs = exps / np.sum(exps, axis=1, keepdims=True)
                    preds = np.argmax(probs, axis=1).tolist()
                    probabilities = probs.tolist()
                else:
                    preds = out_np.squeeze(-1).tolist()
                    if not isinstance(preds, list):
                        preds = [preds]
                return {
                    "predictions": preds,
                    "probabilities": probabilities,
                    "count": len(rows),
                    "explanations": _maybe_explain(),
                }
        except ImportError:
            pass

        # Cualquier modelo con `.predict()` (sklearn, xgboost, lightgbm, catboost, mlflow.pyfunc)
        preds_raw = loaded_model.predict(df)
        preds = preds_raw.tolist() if hasattr(preds_raw, "tolist") else list(preds_raw)
        probabilities = None
        if hasattr(loaded_model, "predict_proba"):
            try:
                probs_raw = loaded_model.predict_proba(df)
                probabilities = (
                    probs_raw.tolist() if hasattr(probs_raw, "tolist") else list(probs_raw)
                )
            except Exception:
                probabilities = None

        return {
            "predictions": preds,
            "probabilities": probabilities,
            "count": len(rows),
            "explanations": _maybe_explain(),
        }


class MLflowPyFuncRuntime(ModelRuntime):
    """Runtime universal basado en MLflow pyfunc para librerías externas personalizadas."""

    runtime_name = "pyfunc"

    def can_handle(self, ml_model: MLModel) -> bool:
        meta = ml_model.metrics_metadata or {}
        return meta.get("framework") in {"pyfunc", "custom", "xgboost", "lightgbm", "catboost", "onnx"}

    def predict(
        self,
        ml_model: MLModel,
        inputs: Union[Dict[str, Any], List[Dict[str, Any]], List[str]],
        pipeline_path: Optional[str] = None,
        explain: bool = False,
    ) -> Dict[str, Any]:
        import mlflow.pyfunc
        from app.platform.tracker.mlflow_service import MLFlowService

        rows = [inputs] if isinstance(inputs, dict) else list(inputs)
        df = prepare_tabular_dataframe(rows, ml_model, pipeline_path) if rows and isinstance(rows[0], dict) else pd.DataFrame(rows)

        cache = get_model_cache()
        cache_key = f"pyfunc:{ml_model.id}:{ml_model.mlflow_run_id}"
        loaded = cache.get(cache_key)
        if loaded is None:
            mlflow_svc = MLFlowService()
            mlflow.set_tracking_uri(mlflow_svc.tracking_uri)
            loaded = mlflow.pyfunc.load_model(f"runs:/{ml_model.mlflow_run_id}/model")
            cache.set(cache_key, loaded, ttl=600)

        preds_raw = loaded.predict(df)
        preds = preds_raw.tolist() if hasattr(preds_raw, "tolist") else list(preds_raw)
        return {"predictions": preds, "probabilities": None, "count": len(rows)}


class ModelRuntimeRegistry:
    """Registro ordenado de runtimes de serving (los más específicos se evalúan primero)."""

    def __init__(self) -> None:
        self._runtimes: Dict[str, ModelRuntime] = {}
        self.register(NLPRuntime())
        self.register(MLflowPyFuncRuntime())
        self.register(TabularMLRuntime())

    def register(self, runtime: ModelRuntime) -> None:
        self._runtimes[runtime.runtime_name] = runtime

    def list_runtimes(self) -> List[str]:
        return list(self._runtimes.keys())

    def resolve(self, ml_model: MLModel) -> ModelRuntime:
        meta = ml_model.metrics_metadata or {}
        explicit = meta.get("serving_runtime")
        if explicit and explicit in self._runtimes:
            return self._runtimes[explicit]
        for runtime in self._runtimes.values():
            if runtime.can_handle(ml_model):
                return runtime
        raise ValueError(f"No hay ModelRuntime compatible para el modelo '{ml_model.id}'.")


_runtime_registry = ModelRuntimeRegistry()


def register_model_runtime(runtime: ModelRuntime) -> None:
    _runtime_registry.register(runtime)


def get_model_runtime_registry() -> ModelRuntimeRegistry:
    return _runtime_registry
