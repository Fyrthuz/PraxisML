"""
Tarea Celery pesada: carga el modelo desde MLFlow, ejecuta la estimación de
incertidumbre, trackea la inferencia en MLFlow y guarda el resultado en disco
y en la base de datos.
"""
import time
import logging
import numpy as np
from datetime import datetime
from io import BytesIO
from pathlib import Path

from celery import Task
from sqlalchemy.orm import Session

from app.worker.celery_app import celery_app
from app.core.config import settings
from app.services.storage_service import get_storage

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="app.worker.tasks.predict.run_heavy_inference")
def run_heavy_inference(
    self: Task,
    tenant_id: str,
    dataset_id: str,
    model_id: str,
    method: str,
    prediction_id: str,
    input_file_path: str = None, # Soporta subida directa de archivo
) -> dict:
    """
    Tarea pesada que NO bloquea la API FastAPI.
    Pipeline completo: load dataset → load model → run estimator →
    log MLFlow inference run → save results → update DB.
    """
    # Lazy imports to avoid circular dependencies at module load time
    from app.database import SessionLocal
    from app.models.dataset import Dataset
    from app.models.ml_model import MLModel
    from app.models.prediction import Prediction
    from app.services.mlflow_service import MLFlowService
    from app.core_ml.factory import PredictionFactory, UncertaintyMethod

    db: Session = SessionLocal()
    storage = get_storage()

    try:
        # ── 1. Marcar como RUNNING ──────────────────────────────────────────
        self.update_state(state="PROGRESS", meta={"status": "Iniciando inferencia..."})
        prediction = db.query(Prediction).filter(Prediction.id == prediction_id).first()
        if not prediction:
            raise ValueError(f"Prediction record {prediction_id} no encontrado en BD.")

        prediction.status = "RUNNING"
        prediction.task_id = self.request.id
        db.commit()

        # ── 2. Cargar Dataset ───────────────────────────────────────────────
        self.update_state(state="PROGRESS", meta={"status": "Cargando dataset/archivo..."})

        input_file_obj = None
        current_file_path = input_file_path
        
        if not current_file_path:
            dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
            if not dataset:
                raise ValueError(f"Dataset {dataset_id} no encontrado en BD.")
            
            # Descargar desde storage
            logger.info(f"Descargando dataset desde storage: {dataset.file_path}")
            data_bytes = storage.download(dataset.file_path)
            input_file_obj = BytesIO(data_bytes)
            current_file_path = dataset.file_path # Para detectar la extensión
        else:
            # Si se pasó una ruta local, la usamos directamente
            input_file_obj = current_file_path

        ext = Path(current_file_path).suffix.lower()
        file_type_str = ext.lstrip(".")

        from app.core_ml.tabular_parser import is_tabular, read_tabular

        if is_tabular(file_type_str):
            input_data = read_tabular(input_file_obj, file_type_str)
        else:
            raise ValueError(f"Formato de archivo '{file_type_str}' no soportado para inferencia tabular.")

        # ── 3. Cargar Modelo ────────────────────────────────────────────────
        self.update_state(state="PROGRESS", meta={"status": "Cargando modelo..."})
        ml_model = db.query(MLModel).filter(MLModel.id == model_id).first()
        if not ml_model:
            raise ValueError(f"Modelo {model_id} no encontrado en BD.")

        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if ml_model.is_torchscript and ml_model.torchscript_path:
            model = torch.jit.load(ml_model.torchscript_path, map_location=device)
        else:
            mlflow_svc = MLFlowService()
            model = mlflow_svc.load_model(ml_model.mlflow_run_id, device=device)

        # ── 4. Ejecutar estimador + trackear en MLFlow ──────────────────────
        self.update_state(state="PROGRESS", meta={"status": f"Ejecutando {method}..."})
        uncertainty_method = UncertaintyMethod(method)
        estimator = PredictionFactory.get_estimator(uncertainty_method, model, device)

        # ── 4b. Apply Preprocessing Pipeline si existe ──────────────────────
        self.update_state(state="PROGRESS", meta={"status": "Aplicando preprocesamiento..."})
        if ml_model.preprocessing_pipeline_path:
            from app.core_ml.preprocessing import load_pipeline, apply_pipeline
            try:
                pipeline = load_pipeline(ml_model.preprocessing_pipeline_path)
                # target_column in inference is usually not present, apply_pipeline handles it
                input_data, _ = apply_pipeline(pipeline, input_data, fit=False)
                # Asegurar formato compatible para el modelo (tensores o numpy floats)
                input_data = input_data.astype(np.float32)
            except Exception as e:
                logger.error("Error aplicando pipeline de preprocesamiento: %s", e)
                raise ValueError(f"Fallo al aplicar pipeline de preprocesamiento: {e}")

        # Recoger hiperparámetros del estimador para loguear en MLFlow
        estimator_params = {
            "uncertainty_method": method,
            "device": str(device),
        }
        if hasattr(estimator, "mc_samples"):
            estimator_params["mc_samples"] = estimator.mc_samples
        if hasattr(estimator, "p"):
            estimator_params["p_dropout"] = estimator.p
        if hasattr(estimator, "tta_samples"):
            estimator_params["tta_samples"] = estimator.tta_samples
        if hasattr(estimator, "n_samples"):
            estimator_params["n_samples"] = estimator.n_samples
        if hasattr(estimator, "noise_std"):
            estimator_params["noise_std"] = estimator.noise_std

        t_start = time.perf_counter()
        results: dict = estimator.estimate_uncertainty(input_data)
        inference_time_s = time.perf_counter() - t_start

        pred_array = results["prediction"]       # Array format depends on data (B, C, H, W) or (N,)
        unc_array = results["uncertainty"]       # Array format depends on data (B, H, W) or (N,)

        # ── 5. Guardar resultados en disco (TODO: también subir a StorageService si se desea) ──
        self.update_state(state="PROGRESS", meta={"status": "Guardando resultados..."})
        results_dir = Path(settings.DATA_DIR) / "tenants" / tenant_id / "predictions" / prediction_id
        results_dir.mkdir(parents=True, exist_ok=True)

        result_path = str(results_dir / "prediction.npy")
        uncertainty_path = str(results_dir / "uncertainty.npy")
        input_data_path = str(results_dir / "input_data.npy")
        np.save(result_path, pred_array)
        np.save(uncertainty_path, unc_array)
        np.save(input_data_path, input_data)

        # ── 6. Loguear run de inferencia en MLFlow (Si no es TorchScript local) ─────────
        self.update_state(state="PROGRESS", meta={"status": "Registrando en MLFlow..."})
        mlflow_inference_run_id = None
        if not ml_model.is_torchscript and ml_model.mlflow_run_id:
            try:
                from app.services.mlflow_service import MLFlowService
                mlflow_svc = MLFlowService()
                with mlflow_svc.start_inference_run(
                    model_run_id=ml_model.mlflow_run_id,
                    tenant_id=tenant_id,
                    method=method,
                    prediction_id=prediction_id,
                ) as run:
                    import mlflow
                    mlflow.log_params(estimator_params)
                    mlflow.log_metric("inference_time_s", inference_time_s)

                    mlflow.log_metrics({
                        "mean_uncertainty": float(np.mean(unc_array)),
                        "max_uncertainty": float(np.max(unc_array)),
                    })
                    mlflow.log_artifact(result_path, artifact_path="results")
                    mlflow.log_artifact(uncertainty_path, artifact_path="results")

                    mlflow_inference_run_id = run.info.run_id
                    print(f"[{method}] Inferencia en MLFlow registrada, Run ID: {mlflow_inference_run_id}")
            except Exception as mlflow_exc:
                logger.warning("MLFlow inference tracking falló (no crítico): %s", mlflow_exc)

        # ── 7. Actualizar registro en BD ────────────────────────────────────
        prediction.status = "COMPLETED"
        prediction.result_path = result_path
        prediction.uncertainty_path = uncertainty_path
        prediction.input_image_path = input_data_path
        prediction.mlflow_inference_run_id = mlflow_inference_run_id
        prediction.completed_at = datetime.utcnow()
        db.commit()

        logger.info("Inferencia completada para prediction_id=%s", prediction_id)
        return {
            "status": "COMPLETED",
            "prediction_id": prediction_id,
            "result_path": result_path,
            "uncertainty_path": uncertainty_path,
            "mlflow_inference_run_id": mlflow_inference_run_id,
            "tenant_id": tenant_id,
        }

    except Exception as exc:
        logger.exception("Error en inferencia prediction_id=%s", prediction_id)
        # Intentar marcar como FAILED en BD
        try:
            if "prediction" in dir():
                prediction.status = "FAILED"
                prediction.error_message = str(exc)
                prediction.completed_at = datetime.utcnow()
                db.commit()
        except Exception:
            pass
        # Re-raise para que Celery marque la tarea como FAILURE
        raise

    finally:
        db.close()
