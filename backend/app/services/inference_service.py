"""
Unified Inference Service.
Consolida la lógica de inferencia de predict.py, single_predict.py y streaming.py.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from app.core.config import settings
from app.services.model_cache import get_model_cache

logger = logging.getLogger(__name__)


class InferenceService:
    """
    Servicio unificado para inferencia de modelos ML.
    Maneja carga de modelos, preprocesamiento e inferencia con incertidumbre.
    """

    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache
        self._cache = get_model_cache() if use_cache else None

    def load_model(
        self,
        ml_model,
        device: Optional[torch.device] = None,
    ) -> torch.nn.Module:
        """
        Carga un modelo desde TorchScript o MLFlow, usando cache si está habilitado.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        is_torchscript = ml_model.is_torchscript
        torchscript_path = ml_model.torchscript_path
        mlflow_run_id = ml_model.mlflow_run_id
        model_id = ml_model.id

        cache_key = f"{model_id}:{is_torchscript}:{torchscript_path or mlflow_run_id}"

        if self._cache:
            model = self._cache.get(cache_key)
            if model is not None:
                logger.info("Modelo %s cargado desde cache", model_id)
                return model

        if is_torchscript and torchscript_path:
            if not Path(torchscript_path).exists():
                raise FileNotFoundError(
                    f"TorchScript file not found: {torchscript_path}"
                )
            logger.info("Cargando TorchScript model desde %s", torchscript_path)
            model = torch.jit.load(torchscript_path, map_location=device)
        else:
            from app.services.mlflow_service import MLFlowService

            mlflow_svc = MLFlowService()
            logger.info("Cargando MLFlow model para run %s", mlflow_run_id)
            model = mlflow_svc.load_model(mlflow_run_id, device=device)

        if model is None:
            raise ValueError(f"No se pudo cargar el modelo {model_id}")

        if hasattr(model, "to"):
            model.to(device)
        if hasattr(model, "eval"):
            model.eval()

        if self._cache:
            self._cache.set(cache_key, model, ttl=600)
            logger.info("Modelo %s almacenado en cache", model_id)

        return model

    def load_preprocessing_pipeline(self, pipeline_path: str):
        """Carga el pipeline de preprocesamiento si existe."""
        if not pipeline_path:
            return None

        try:
            from app.core_ml.preprocessing import load_pipeline

            logger.info("Cargando pipeline desde %s", pipeline_path)
            return load_pipeline(pipeline_path)
        except Exception as e:
            logger.error("Error cargando pipeline de preprocesamiento: %s", str(e))
            return None

    def preprocess_features(
        self,
        features: Dict[str, Any],
        feature_names: list,
        pipeline: Optional[Any] = None,
    ) -> np.ndarray:
        """Convierte features dict a numpy array para inferencia."""
        import pandas as pd

        df = pd.DataFrame([features])

        if feature_names:
            for col in feature_names:
                if col not in df.columns:
                    df[col] = 0.0
            df = df[feature_names]

        if pipeline is not None:
            from app.core_ml.preprocessing import apply_pipeline

            df, _ = apply_pipeline(pipeline, df)

        return df.to_numpy().astype(np.float32)

    def run_inference(
        self,
        model: torch.nn.Module,
        input_data: np.ndarray,
        method: str = "ensemble",
        device: Optional[torch.device] = None,
        mc_samples: int = 100,
        tta_samples: int = 10,
    ) -> Dict[str, Any]:
        """
        Ejecuta inferencia con cálculo de incertidumbre.
        """
        if device is None:
            device = next(model.parameters()).device

        from app.core_ml.factory import PredictionFactory, UncertaintyMethod

        unc_method = UncertaintyMethod(method.lower())
        estimator = PredictionFactory.get_estimator(
            method=unc_method,
            model=model,
            device=device,
            mc_samples=mc_samples,
            tta_samples=tta_samples,
        )

        result = estimator.estimate_uncertainty(input_data)
        return result

    def run_inference_with_shap(
        self,
        model: torch.nn.Module,
        input_data: np.ndarray,
        feature_names: list,
        method: str = "ensemble",
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """Ejecuta inferencia incluyendo SHAP values."""
        result = self.run_inference(model, input_data, method, device)

        try:
            from app.core_ml.explainability import get_shap_values

            shap_result = get_shap_values(model, input_data, feature_names)
            result["shap_values"] = shap_result["shap_values"]
            result["shap_expected_value"] = shap_result["expected_value"]
            result["feature_names"] = feature_names
        except Exception as e:
            logger.warning("No se pudieron calcular SHAP values: %s", str(e))

        return result

    def save_prediction(
        self,
        tenant_id: str,
        prediction_id: str,
        prediction_data: np.ndarray,
        uncertainty_data: Optional[np.ndarray] = None,
        input_data: Optional[np.ndarray] = None,
    ) -> Tuple[str, Optional[str], Optional[str]]:
        """Guarda los resultados de la predicción en disco."""
        out_dir = Path(settings.DATA_DIR) / "tenants" / tenant_id / "predictions"
        out_dir.mkdir(parents=True, exist_ok=True)

        pred_path = out_dir / f"{prediction_id}_pred.npy"
        np.save(str(pred_path), prediction_data)

        unc_path = None
        if uncertainty_data is not None:
            unc_path = out_dir / f"{prediction_id}_unc.npy"
            np.save(str(unc_path), uncertainty_data)

        in_path = None
        if input_data is not None:
            in_path = out_dir / f"{prediction_id}_input.npy"
            np.save(str(in_path), input_data)

        return (
            str(pred_path),
            str(unc_path) if unc_path else None,
            str(in_path) if in_path else None,
        )


_inference_service_instance: Optional[InferenceService] = None


def get_inference_service(use_cache: bool = True) -> InferenceService:
    """Factory para obtener una instancia del InferenceService."""
    global _inference_service_instance
    if _inference_service_instance is None:
        _inference_service_instance = InferenceService(use_cache=use_cache)
    return _inference_service_instance
