"""
Tarea Celery para inferencia NLP en batch sobre CSV.

Lee el CSV, predice cada fila con el modelo NLP del tenant y guarda
los resultados como JSON. Actualiza el registro Prediction.
"""

import json
import logging
from pathlib import Path

from celery import Task
from sqlalchemy.orm import Session

from app.core.config import settings
from app.worker.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="app.worker.tasks.predict_nlp.run_nlp_batch_inference")
def run_nlp_batch_inference(
    self: Task,
    tenant_id: str,
    model_id: str,
    prediction_id: str,
    input_file_path: str = None,
    text_column: str = "text",
):
    """
    Inferencia NLP batch sobre un CSV.

    Args:
        input_file_path: ruta local temporal del CSV subido.
    """
    from app.database import SessionLocal
    from app.models.ml_model import MLModel
    from app.models.prediction import Prediction

    db: Session = SessionLocal()
    try:
        prediction = db.query(Prediction).filter(Prediction.id == prediction_id).first()
        if prediction:
            prediction.status = "RUNNING"
            db.commit()

        ml_model = (
            db.query(MLModel)
            .filter(MLModel.id == model_id, MLModel.tenant_id == tenant_id)
            .first()
        )
        if not ml_model:
            raise ValueError(f"Modelo {model_id} no encontrado para tenant {tenant_id}")

        from app.services.nlp_inference import get_nlp_inference_service

        service = get_nlp_inference_service()
        handle = service.load_model(ml_model)

        if input_file_path:
            import pandas as pd

            df = pd.read_csv(input_file_path)
            if text_column not in df.columns:
                raise ValueError(f"Columna '{text_column}' no encontrada en el CSV")
        else:
            raise ValueError("input_file_path es obligatorio para batch NLP")

        results = service.predict_batch(handle, df, text_column)

        out_dir = Path(settings.DATA_DIR) / "tenants" / tenant_id / "nlp_predictions"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{prediction_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False)

        metadata = ml_model.metrics_metadata or {}
        total_tokens = sum(len(r.get("tokens", [])) for r in results)
        try:
            service.record_usage(
                db=db,
                tenant_id=tenant_id,
                model_id=model_id,
                tokens_in=total_tokens,
                latency_ms=0,
                tags={
                    "algorithm": metadata.get("algorithm", ""),
                    "task_type": metadata.get("task_type", "text_classification"),
                    "batch": "true",
                },
            )
        except Exception as exc:
            logger.warning("Usage recording failed (non-critical): %s", exc)

        if prediction:
            prediction.status = "COMPLETED"
            prediction.result_path = str(out_path)
            prediction.error_message = None
            db.commit()

        return {
            "status": "COMPLETED",
            "prediction_id": prediction_id,
            "result_path": str(out_path),
            "n_results": len(results),
        }

    except Exception as exc:
        logger.error("Error en batch NLP: %s", exc, exc_info=True)
        try:
            prediction = db.query(Prediction).filter(Prediction.id == prediction_id).first()
            if prediction:
                prediction.status = "FAILED"
                prediction.error_message = str(exc)
                db.commit()
        except Exception:
            pass
        raise exc
    finally:
        db.close()
