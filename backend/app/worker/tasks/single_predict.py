import logging
import torch
import torch.nn as nn
import mlflow
from celery import Task
from pathlib import Path

from app.worker.celery_app import celery_app
from app.core_ml.factory import PredictionFactory, UncertaintyMethod

logger = logging.getLogger(__name__)

@celery_app.task(bind=True, name="app.worker.tasks.single_predict.run_single_tabular_inference")
def run_single_tabular_inference(
    self: Task,
    tenant_id: str,
    image_path: str, # Mantener por compatibilidad de firma si es necesario, aunque sea None
    model_id: str,
    method: str,
    prediction_id: str,
    features: dict = None,
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
        
        if ml_model.is_torchscript and ml_model.torchscript_path:
            logger.info("Cargando TorchScript model...")
            model = torch.jit.load(ml_model.torchscript_path, map_location=device)
        else:
            logger.info("Cargando MLFlow model...")
            mlflow_svc = MLFlowService()
            model = mlflow_svc.load_model(ml_model.mlflow_run_id, device=device)
            
        model.to(device)
        model.eval()
        
        # 3. Process features
        if features is not None:
            import pandas as pd
            feature_names = ml_model.metrics_metadata.get("feature_names", [])
            df = pd.DataFrame([features])
            
            if feature_names:
                for col in feature_names:
                    if col not in df.columns:
                        df[col] = 0.0
                df = df[feature_names]
            
            # Apply preprocessing if available
            if ml_model.preprocessing_pipeline_path:
                from app.core_ml.preprocessing import load_pipeline, apply_pipeline
                try:
                    pipeline = load_pipeline(ml_model.preprocessing_pipeline_path)
                    df, _ = apply_pipeline(pipeline, df)
                except Exception as e:
                    logger.error(f"Error applying preprocessing to single inference: {e}")
                    raise ValueError(f"Fallo al aplicar pipeline de preprocesamiento: {e}")

            input_data = df.to_numpy().astype(np.float32)
        else:
            raise ValueError("No se proporcionaron features para la inferencia tabular.")
        
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
        
        # 5. Guardar predicciones numpy 
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
