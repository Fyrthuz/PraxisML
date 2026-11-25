"""
Tarea Celery para entrenamiento de modelos NLP (HuggingFace Transformers).

Espejo de `app.worker.tasks.train.run_training` para el dominio NLP:
carga el dataset, entrena con NLPTrainer, registra el MLModel en BD,
registra observabilidad (uso + coste) e indexa el embedding del modelo.
"""

import logging

from celery import Task
from sqlalchemy.orm import Session

from app.core.config import settings
from app.worker.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="app.worker.tasks.train_nlp.run_nlp_training")
def run_nlp_training(
    self: Task,
    tenant_id: str,
    dataset_id: str,
    text_column: str,
    label_column: str,
    algorithm: str,
    task_type: str,
    hyperparams: dict,
    validation_config: dict,
    model_name: str,
    model_description: str,
    registry_name: str = None,
):
    """
    Pipeline completo de entrenamiento NLP:
        1. Cargar dataset tabular desde BD/storage
        2. Entrenar con NLPTrainer (HF Transformers + MLflow autolog + ONNX)
        3. Registrar modelo en la tabla ml_model (+ Registry opcional)
        4. Observabilidad: usage + coste (dominio nlp)
        5. Indexar embedding del modelo (semantic search, best-effort)
    """
    from io import BytesIO

    from app.core_ml.hyperparams import get_nlp_algorithm_info
    from app.core_ml.tabular_parser import read_tabular
    from app.database import SessionLocal
    from app.models.dataset import Dataset
    from app.models.ml_model import MLModel

    db: Session = SessionLocal()

    try:
        # 1. Cargar dataset
        logger.info("Cargando dataset %s para entrenamiento NLP...", dataset_id)
        dataset = (
            db.query(Dataset)
            .filter(Dataset.id == dataset_id, Dataset.tenant_id == tenant_id)
            .first()
        )
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} no encontrado.")

        from app.platform.storage.service import get_storage

        storage = get_storage()
        data_bytes = storage.download(dataset.file_path)
        df = read_tabular(BytesIO(data_bytes), dataset.file_type)
        logger.info(
            "Dataset cargado: %d filas × %d columnas", len(df), len(df.columns)
        )

        # 2. Entrenar
        algo_info = get_nlp_algorithm_info(algorithm)
        final_model_name = model_name or (
            f"{algo_info['display_name']} — {dataset.name}"
        )

        from app.domains.nlp.trainer import NLPTrainer

        trainer = NLPTrainer(
            tenant_id=tenant_id,
            mlflow_tracking_uri=settings.MLFLOW_TRACKING_URI or None,
        )
        result = trainer.train(
            df=df,
            text_column=text_column,
            label_column=label_column,
            algorithm=algorithm,
            task_type=task_type,
            hyperparams=hyperparams,
            validation_config=validation_config,
            model_name=final_model_name,
        )

        metrics = result["metrics"]
        mlflow_run_id = result["mlflow_run_id"]
        label_map = result["label_map"]
        id2label = result.get("id2label") or {str(v): k for k, v in (label_map or {}).items()}
        onnx_path = result.get("onnx_path")
        model_path = result.get("model_path")
        tokenizer_path = result.get("tokenizer_path")

        logger.info(
            "Entrenamiento NLP completado. run_id=%s metrics=%s",
            mlflow_run_id,
            metrics,
        )

        # 3. Registry opcional
        mlflow_registry_name = None
        mlflow_version = None
        if registry_name:
            try:
                from app.platform.tracker.mlflow_service import MLFlowService

                mlflow_svc = MLFlowService()
                try:
                    mlflow_svc.create_registered_model(
                        name=registry_name,
                        description=model_description or f"Model: {final_model_name}",
                    )
                except Exception:
                    pass  # Ya existe

                registry_result = mlflow_svc.register_model_to_registry(
                    model_name=registry_name,
                    run_id=mlflow_run_id,
                    description=model_description or "",
                )
                mlflow_registry_name = registry_result["name"]
                mlflow_version = str(registry_result["version"])
            except Exception as e:
                logger.warning("No se pudo registrar en MLflow Registry: %s", e)

        # 4. Registrar modelo en BD
        new_model = MLModel(
            name=final_model_name,
            description=model_description
            or f"Trained on '{dataset.name}' ({task_type})",
            mlflow_run_id=mlflow_run_id,
            mlflow_registry_name=mlflow_registry_name,
            mlflow_version=mlflow_version,
            preprocessing_pipeline_path=dataset.pipeline_path,
            metrics_metadata={
                "framework": "huggingface",
                "domain": "nlp",
                "algorithm": algorithm,
                "task_type": task_type,
                "text_column": text_column,
                "label_column": label_column,
                "label_map": label_map,
                "id2label": id2label,
                "model_path": model_path,
                "tokenizer_path": tokenizer_path,
                "onnx_path": onnx_path,
                "metrics": metrics,
                "feature_names": [text_column],
                "dataset_id": dataset_id,
                "dataset_name": dataset.name,
                **{
                    hp_name: hp_val
                    for hp_name, hp_val in (hyperparams or {}).items()
                    if not isinstance(hp_val, (list, dict))
                },
            },
            is_active=True,
            is_public=False,
            tenant_id=tenant_id,
            dataset_dvc_hash=dataset.dvc_hash,
            dataset_dvc_registry_name=dataset.dvc_registry_name,
        )
        db.add(new_model)
        db.commit()
        db.refresh(new_model)

        logger.info("Modelo NLP registrado en BD: id=%s", new_model.id)

        # 5. Observabilidad
        try:
            from datetime import datetime, timezone

            from app.observability.cost_calculator import get_cost_calculator
            from app.observability.usage_collector import get_usage_collector

            collector = get_usage_collector()
            calculator = get_cost_calculator()

            duration_s = result.get("duration_s", 0)
            on_gpu = result.get("device") == "cuda"
            collector.record_training(
                db=db,
                tenant_id=tenant_id,
                model_id=new_model.id,
                domain="nlp",
                duration_s=duration_s,
                cpu_hours=duration_s / 3600 if not on_gpu else 0,
                gpu_hours=duration_s / 3600 if on_gpu else 0,
                metrics=metrics,
                tags={
                    "algorithm": algorithm,
                    "task_type": task_type,
                    "framework": "huggingface",
                },
            )

            cost = calculator.estimate_training_cost(
                db=db,
                gpu_hours=duration_s / 3600 if on_gpu else 0,
                gpu_type="A100" if on_gpu else None,
                cpu_hours=duration_s / 3600 if not on_gpu else 0,
            )
            if cost > 0:
                now = datetime.now(timezone.utc)
                calculator.record_cost(
                    db=db,
                    tenant_id=tenant_id,
                    cost_type="training",
                    amount=cost,
                    source=mlflow_run_id,
                    period_start=now,
                    period_end=now,
                    model_id=new_model.id,
                    metadata={"algorithm": algorithm, "domain": "nlp"},
                )
            db.commit()
        except Exception as obs_exc:
            logger.warning("Observability recording failed (non-critical): %s", obs_exc)
            try:
                db.rollback()
            except Exception:
                pass

        # 6. Indexar embedding (best-effort)
        try:
            from app.domains.nlp.embedding_service import get_embedding_service

            get_embedding_service().index_model(db, new_model)
            db.commit()
        except Exception as index_exc:
            logger.warning("Embedding index failed (non-critical): %s", index_exc)
            try:
                db.rollback()
            except Exception:
                pass

        return {
            "status": "COMPLETED",
            "model_id": new_model.id,
            "mlflow_run_id": mlflow_run_id,
            "metrics": metrics,
            "algorithm": algorithm,
            "framework": "huggingface",
            "task_type": task_type,
            "dataset_info": {
                "id": dataset.id,
                "name": dataset.name,
                "version": dataset.version,
            },
        }

    except Exception as exc:
        logger.error("Error en entrenamiento NLP: %s", exc, exc_info=True)
        raise exc
    finally:
        db.close()
