"""
Tarea Celery para entrenamiento de modelos sklearn y PyTorch.
No bloquea la API — el frontend hace polling del estado.
"""

import logging

from celery import Task
from sqlalchemy.orm import Session

from app.core.config import settings
from app.worker.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="app.worker.tasks.train.run_training")
def run_training(
    self: Task,
    tenant_id: str,
    dataset_id: str,
    target_column: str,
    algorithm: str,
    task_type: str,
    hyperparams: dict,
    validation_config: dict,
    model_name: str,
    model_description: str,
    registry_name: str = None,
):
    """
    Pipeline completo de entrenamiento:
        1. Detectar framework (sklearn vs pytorch) según el algoritmo
        2. Cargar dataset tabular desde BD/disco
        3. Entrenar modelo con el trainer correspondiente (+ MLFlow autolog)
        4. Registrar modelo en la tabla ml_model
        5. Devolver métricas y run_id
    """
    from app.core_ml.hyperparams import get_algorithm_info
    from app.core_ml.tabular_parser import read_tabular
    from app.database import SessionLocal
    from app.models.dataset import Dataset
    from app.models.ml_model import MLModel

    db: Session = SessionLocal()

    try:
        # 0. Detectar framework
        algo_info = get_algorithm_info(algorithm)
        framework = algo_info.get("framework", "sklearn")

        # 1. Cargar dataset
        logger.info("Cargando dataset %s para entrenamiento...", dataset_id)
        dataset = (
            db.query(Dataset)
            .filter(Dataset.id == dataset_id, Dataset.tenant_id == tenant_id)
            .first()
        )
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} no encontrado.")

        from io import BytesIO

        from app.services.storage_service import get_storage
        storage = get_storage()
        data_bytes = storage.download(dataset.file_path)
        df = read_tabular(BytesIO(data_bytes), dataset.file_type)
        logger.info("Dataset cargado desde storage: %d filas × %d columnas", len(df), len(df.columns))

        # 2. Preparar hiperparámetros especiales para PyTorch
        if framework == "pytorch":
            # Convertir hidden_layers de string "128,64" a lista [128, 64]
            if "hidden_layers" in hyperparams and isinstance(
                hyperparams["hidden_layers"], str
            ):
                hyperparams["hidden_layers"] = [
                    int(x.strip())
                    for x in hyperparams["hidden_layers"].split(",")
                    if x.strip()
                ]

        # 3. Entrenar según framework
        logger.info(
            "Entrenando: framework=%s, algorithm=%s, target=%s, task=%s",
            framework,
            algorithm,
            target_column,
            task_type,
        )

        final_model_name = model_name or f"{algo_info['display_name']} — {dataset.name}"

        mlflow_uri = settings.MLFLOW_TRACKING_URI or None

        if framework == "pytorch":
            from app.services.training_service import PyTorchTrainer

            trainer = PyTorchTrainer(
                tenant_id=tenant_id, mlflow_tracking_uri=mlflow_uri
            )
        else:
            from app.services.training_service import SklearnTrainer

            trainer = SklearnTrainer(
                tenant_id=tenant_id, mlflow_tracking_uri=mlflow_uri
            )

        result = trainer.train(
            df=df,
            target_column=target_column,
            algorithm=algorithm,
            task_type=task_type,
            hyperparams=hyperparams,
            validation_config=validation_config,
            model_name=final_model_name,
        )

        metrics = result["metrics"]
        mlflow_run_id = result["mlflow_run_id"]
        model_path = result["model_path"]
        feature_names = result["feature_names"]

        logger.info(
            "Entrenamiento completado. run_id=%s metrics=%s", mlflow_run_id, metrics
        )

        # 4. Optionally register to MLflow Model Registry
        mlflow_registry_name = None
        mlflow_version = None
        if registry_name:
            try:
                from app.services.mlflow_service import MLFlowService

                mlflow_svc = MLFlowService()

                # Create registry if it doesn't exist (or use existing)
                try:
                    mlflow_svc.create_registered_model(
                        name=registry_name,
                        description=model_description or f"Model: {final_model_name}",
                    )
                except Exception:
                    pass  # Already exists

                # Register model version
                registry_result = mlflow_svc.register_model_to_registry(
                    model_name=registry_name,
                    run_id=mlflow_run_id,
                    description=model_description or "",
                )
                mlflow_registry_name = registry_result["name"]
                mlflow_version = str(registry_result["version"])
                logger.info(
                    "Modelo registrado en MLflow Registry: %s v%s",
                    mlflow_registry_name,
                    mlflow_version,
                )
            except Exception as e:
                logger.warning("No se pudo registrar en MLflow Registry: %s", e)

        # 5. Registrar modelo en BD
        new_model = MLModel(
            name=final_model_name,
            description=model_description
            or f"Trained on '{dataset.name}' ({task_type})",
            mlflow_run_id=mlflow_run_id,
            mlflow_registry_name=mlflow_registry_name,
            mlflow_version=mlflow_version,
            preprocessing_pipeline_path=dataset.pipeline_path,
            metrics_metadata={
                "framework": framework,
                "algorithm": algorithm,
                "task_type": task_type,
                "target_column": target_column,
                "metrics": metrics,
                "feature_names": feature_names,
                "dataset_id": dataset_id,
                "dataset_name": dataset.name,
                **{
                    hp_name: hp_val
                    for hp_name, hp_val in hyperparams.items()
                    if not isinstance(hp_val, (list, dict))
                },
            },
            is_active=True,
            is_public=False,
            tenant_id=tenant_id,
            # Trazabilidad DVC
            dataset_dvc_hash=dataset.dvc_hash,
            dataset_dvc_registry_name=dataset.dvc_registry_name,
        )
        db.add(new_model)
        db.commit()
        db.refresh(new_model)

        logger.info("Modelo registrado en BD: id=%s", new_model.id)

        return {
            "status": "COMPLETED",
            "model_id": new_model.id,
            "mlflow_run_id": mlflow_run_id,
            "model_path": model_path,
            "metrics": metrics,
            "algorithm": algorithm,
            "framework": framework,
            "task_type": task_type,
            "dataset_info": {
                "id": dataset.id,
                "name": dataset.name,
                "version": dataset.version,
                "dvc_hash": dataset.dvc_hash,
                "dvc_registry_name": dataset.dvc_registry_name,
                "is_dvc_tracked": dataset.is_dvc_tracked,
            },
        }

    except Exception as exc:
        logger.error("Error en entrenamiento: %s", exc, exc_info=True)
        raise exc
    finally:
        db.close()
