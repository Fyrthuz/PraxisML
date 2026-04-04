import logging
import torch
from celery import Task
from pathlib import Path

from app.worker.celery_app import celery_app
from app.core_ml.factory import PredictionFactory, UncertaintyMethod
from app.services.inference_service import get_inference_service

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True, name="app.worker.tasks.single_predict.run_single_tabular_inference"
)
def run_single_tabular_inference(
    self: Task,
    tenant_id: str,
    image_path: str,  # Mantener por compatibilidad de firma si es necesario, aunque sea None
    model_id: str,
    method: str,
    prediction_id: str,
    features: dict = None,
    explain: bool = False,
):
    """
    Tarea Celery liviana:
    1. Carga un modelo TorchScript de la DB o MLFlow.
    2. Procesa las características tabulares (JSON) a numpy.
    3. Pasa por el Estimator.
    4. Guarda resultado y actualiza DB.
    """
    from app.database import SessionLocal
    from app.models.prediction import Prediction
    from app.models.ml_model import MLModel
    from app.services.mlflow_service import MLFlowService
    import numpy as np
    from app.core.config import settings

    logger.info(f"[{prediction_id}] Iniciando Single Tabular Inference: {method}")
    db = SessionLocal()

    try:
        # 1. Update status
        prediction = db.query(Prediction).filter(Prediction.id == prediction_id).first()
        prediction.status = "IN_PROGRESS"
        db.commit()

        # 2. Load Model
        ml_model = db.query(MLModel).filter(MLModel.id == model_id).first()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        inference_svc = get_inference_service(use_cache=True)
        model = inference_svc.load_model(ml_model, device=device)

        # 3. Process features
        if features is not None:
            import pandas as pd

            feature_names = ml_model.metrics_metadata.get("feature_names", [])
            pipeline = None
            if ml_model.preprocessing_pipeline_path:
                pipeline = inference_svc.load_preprocessing_pipeline(
                    ml_model.preprocessing_pipeline_path
                )

            input_data = inference_svc.preprocess_features(
                features, feature_names, pipeline
            )
        else:
            raise ValueError(
                "No se proporcionaron features para la inferencia tabular."
            )

        # 4. Infer via Factory
        unc_method = UncertaintyMethod(method.lower())
        estimator = PredictionFactory.get_estimator(
            method=unc_method,
            model=model,
            device=device,
            # Custom defaults for rapid single test
            mc_samples=5,
            tta_samples=5,
        )

        # El input a inferir debe ser numpy
        result_dict = estimator.estimate_uncertainty(input_data)

        # 5. Calcular SHAP values si se solicita
        if explain:
            from app.core_ml.explainability import get_shap_values

            try:
                feature_names = ml_model.metrics_metadata.get("feature_names", [])
                if not feature_names:
                    feature_names = [f"feature_{i}" for i in range(input_data.shape[1])]

                shap_result = get_shap_values(model, input_data, feature_names)
                result_dict["shap_values"] = shap_result["shap_values"]
                result_dict["shap_expected_value"] = shap_result["expected_value"]
                result_dict["feature_names"] = feature_names
            except Exception as e:
                logger.warning(f"No se pudieron calcular SHAP values: {e}")

        # 6. Guardar predicciones numpy
        out_dir = Path(settings.DATA_DIR) / "tenants" / tenant_id / "predictions"
        out_dir.mkdir(parents=True, exist_ok=True)

        pred_path = out_dir / f"{prediction_id}_pred.npy"
        unc_path = out_dir / f"{prediction_id}_unc.npy"
        in_path = out_dir / f"{prediction_id}_input.npy"

        np.save(str(pred_path), result_dict["prediction"])
        np.save(str(in_path), input_data)
        if "uncertainty" in result_dict:
            np.save(str(unc_path), result_dict["uncertainty"])
            prediction.uncertainty_path = str(unc_path)

        prediction.result_path = str(pred_path)
        prediction.input_image_path = str(in_path)
        prediction.status = "COMPLETED"
        db.commit()
        logger.info(f"[{prediction_id}] Exito Single Tabular Inference")

        return "COMPLETED"

    except Exception as e:
        logger.error(f"[{prediction_id}] Failed: {str(e)}", exc_info=True)
        prediction = db.query(Prediction).filter(Prediction.id == prediction_id).first()
        if prediction:
            prediction.status = "FAILED"
            prediction.error_message = str(e)
            db.commit()
        raise e
    finally:
        db.close()
