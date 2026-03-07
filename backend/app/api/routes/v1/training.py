"""
API route para entrenamiento de modelos sklearn.
"""
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from typing import Any, Dict, Optional
from pydantic import BaseModel
import logging

from app.database import get_db
from app.models.user import User
from app.models.tenant import Tenant
from app.api.deps import (
    require_editor,
    require_viewer,
    check_training_quota,
)
from app.core_ml.hyperparams import get_all_algorithms, get_algorithm_info
from app.core.rate_limit import limiter
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Schemas ──────────────────────────────────────────────────────────────────

class ValidationConfig(BaseModel):
    strategy: str = "holdout"        # "holdout" or "cross_validation"
    test_size: float = 0.2           # holdout: proportion for test set
    n_folds: int = 5                 # cross_validation: number of folds
    shuffle: bool = True
    random_state: int = 42


class TrainRequest(BaseModel):
    dataset_id: str
    target_column: str
    algorithm: str
    task_type: str = "classification"
    hyperparams: Dict[str, Any] = {}
    validation: ValidationConfig = ValidationConfig()
    model_name: Optional[str] = None
    model_description: Optional[str] = None


class TrainResponse(BaseModel):
    message: str
    task_id: str
    status_url: str


# ──────────────────────────────────────────────────────────────────────────────
# GET /training/algorithms — Listar algoritmos disponibles
# ──────────────────────────────────────────────────────────────────────────────
@router.get("/algorithms")
def list_algorithms(_user: User = Depends(require_viewer)):
    """
    Devuelve la lista de algoritmos disponibles con sus hiperparámetros configurables.
    El frontend usa esto para renderizar formularios dinámicos.
    Requiere rol **viewer** o superior.
    """
    return get_all_algorithms()


# ──────────────────────────────────────────────────────────────────────────────
# POST /training/train — Lanzar entrenamiento
# ──────────────────────────────────────────────────────────────────────────────
@router.post("/train", response_model=TrainResponse)
@limiter.limit(settings.RATE_LIMIT_TRAINING)
def start_training(
    request: Request,
    req: TrainRequest,
    _user: User = Depends(require_editor),
    tenant: Tenant = Depends(check_training_quota),
    db: Session = Depends(get_db),
):
    """
    Lanza una tarea Celery de entrenamiento para un dataset tabular.
    No bloquea — devuelve task_id para polling.

    Requiere rol **editor** o superior. Rate limited a {RATE_LIMIT_TRAINING}.
    Valida cuota diaria de entrenamientos del tenant.
    """
    # Validar que el algoritmo existe
    try:
        algo_info = get_algorithm_info(req.algorithm)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Validar task_type
    if req.task_type not in algo_info["task_types"]:
        raise HTTPException(
            status_code=400,
            detail=f"El algoritmo '{req.algorithm}' no soporta task_type='{req.task_type}'. "
            f"Tipos soportados: {algo_info['task_types']}",
        )

    # Validar que el dataset existe
    from app.models.dataset import Dataset
    dataset = db.query(Dataset).filter(
        Dataset.id == req.dataset_id, Dataset.tenant_id == tenant.id
    ).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset no encontrado.")

    # Validar que es tabular
    from app.core_ml.tabular_parser import is_tabular
    if not dataset.file_type or not is_tabular(dataset.file_type):
        raise HTTPException(
            status_code=400,
            detail="El entrenamiento solo es posible con datasets tabulares (.csv, .xlsx, .parquet).",
        )

    # Validar que tiene preprocesamiento asociado
    if not dataset.pipeline_path:
        raise HTTPException(
            status_code=400,
            detail="El dataset seleccionado no tiene un preprocesamiento (pipeline) asociado. Por favor, aplica un preprocesamiento antes de entrenar.",
        )

    # Validar que target_column existe en el dataset
    if dataset.column_names and req.target_column not in dataset.column_names:
        raise HTTPException(
            status_code=400,
            detail=f"La columna '{req.target_column}' no existe en el dataset. "
            f"Columnas disponibles: {dataset.column_names}",
        )

    # Lanzar tarea Celery
    from app.worker.tasks.train import run_training
    task = run_training.delay(
        tenant_id=tenant.id,
        dataset_id=req.dataset_id,
        target_column=req.target_column,
        algorithm=req.algorithm,
        task_type=req.task_type,
        hyperparams=req.hyperparams,
        validation_config=req.validation.model_dump(),
        model_name=req.model_name or f"{algo_info['display_name']} — {dataset.name}",
        model_description=req.model_description or "",
    )

    return TrainResponse(
        message="Entrenamiento lanzado correctamente.",
        task_id=task.id,
        status_url=f"/api/v1/training/status/{task.id}",
    )


# ──────────────────────────────────────────────────────────────────────────────
# GET /training/status/{task_id} — Polling de estado
# ──────────────────────────────────────────────────────────────────────────────
@router.get("/status/{task_id}")
def get_training_status(
    task_id: str,
    _user: User = Depends(require_viewer),
):
    """
    Consulta el estado de una tarea de entrenamiento Celery.
    Requiere rol **viewer** o superior.
    """
    from celery.result import AsyncResult
    from app.worker.celery_app import celery_app

    result = AsyncResult(task_id, app=celery_app)

    response = {
        "task_id": task_id,
        "status": result.status,
    }

    if result.ready():
        if result.successful():
            response["result"] = result.result
        else:
            response["error"] = str(result.result) if result.result else "Unknown error"

    return response
