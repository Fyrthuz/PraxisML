"""
API route para entrenamiento de modelos NLP (HuggingFace Transformers).

Endpoints:
  GET  /api/v1/nlp/algorithms          → Arquitecturas disponibles
  POST /api/v1/nlp/train               → Lanzar entrenamiento NLP (Celery)
  GET  /api/v1/nlp/training/status/{id} → Estado del entrenamiento
"""

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import (
    check_training_quota,
    require_editor,
    require_viewer,
)
from app.core.config import settings
from app.core.rate_limit import limiter
from app.core_ml.hyperparams import (
    get_all_nlp_algorithms,
    get_nlp_algorithm_info,
)
from app.database import get_db
from app.models.tenant import Tenant
from app.models.user import User

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Schemas ──────────────────────────────────────────────────────────────────


class ValidationConfig(BaseModel):
    strategy: str = "holdout"
    test_size: float = 0.2
    n_folds: int = 5
    shuffle: bool = True
    random_state: int = 42


class TrainRequest(BaseModel):
    dataset_id: str
    text_column: str
    label_column: Optional[str] = None
    algorithm: str = "distilbert"
    task_type: str = "text_classification"
    hyperparams: Dict[str, Any] = {}
    validation: ValidationConfig = ValidationConfig()
    model_name: Optional[str] = None
    model_description: Optional[str] = None
    registry_name: Optional[str] = None


class TrainResponse(BaseModel):
    message: str
    task_id: str
    status_url: str


# ──────────────────────────────────────────────────────────────────────────────
# GET /algorithms — Listar arquitecturas NLP
# ──────────────────────────────────────────────────────────────────────────────


@router.get("/algorithms")
def list_nlp_algorithms(_user: User = Depends(require_viewer)):
    """
    Devuelve las arquitecturas NLP disponibles (DistilBERT, BERT, RoBERTa, ...).
    El frontend usa esto para renderizar formularios dinámicos.
    """
    return get_all_nlp_algorithms()


# ──────────────────────────────────────────────────────────────────────────────
# POST /train — Lanzar entrenamiento NLP
# ──────────────────────────────────────────────────────────────────────────────


@router.post("/train", response_model=TrainResponse)
@limiter.limit(settings.RATE_LIMIT_TRAINING)
def start_nlp_training(
    request: Request,
    req: TrainRequest,
    _user: User = Depends(require_editor),
    tenant: Tenant = Depends(check_training_quota),
    db: Session = Depends(get_db),
):
    """
    Lanza una tarea Celery de entrenamiento NLP para un dataset textual.
    No bloquea — devuelve task_id para polling.

    Requiere rol **editor** o superior. Rate limited a {RATE_LIMIT_TRAINING}.
    Valida cuota diaria de entrenamientos del tenant.
    """
    # Validar algoritmo
    try:
        algo_info = get_nlp_algorithm_info(req.algorithm)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Validar task_type
    if req.task_type not in algo_info["task_types"]:
        raise HTTPException(
            status_code=400,
            detail=f"El algoritmo '{req.algorithm}' no soporta task_type='{req.task_type}'. "
            f"Tipos soportados: {algo_info['task_types']}",
        )

    # Validar dataset
    from app.models.dataset import Dataset

    dataset = (
        db.query(Dataset)
        .filter(Dataset.id == req.dataset_id, Dataset.tenant_id == tenant.id)
        .first()
    )
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset no encontrado.")

    # Validar que es tabular
    from app.core_ml.tabular_parser import is_tabular

    if not dataset.file_type or not is_tabular(dataset.file_type):
        raise HTTPException(
            status_code=400,
            detail="El entrenamiento NLP requiere datasets tabulares (.csv, .xlsx, .parquet).",
        )

    # Validar columnas
    if dataset.column_names:
        if req.text_column not in dataset.column_names:
            raise HTTPException(
                status_code=400,
                detail=f"La columna de texto '{req.text_column}' no existe en el dataset. "
                f"Columnas disponibles: {dataset.column_names}",
            )
        if req.label_column and req.label_column not in dataset.column_names:
            raise HTTPException(
                status_code=400,
                detail=f"La columna objetivo '{req.label_column}' no existe en el dataset. "
                f"Columnas disponibles: {dataset.column_names}",
            )

    # Lanzar tarea Celery
    from app.worker.tasks.train_nlp import run_nlp_training

    registry_name = req.registry_name
    if registry_name and not registry_name.startswith(f"tenant_{tenant.id}_"):
        registry_name = f"tenant_{tenant.id}_{registry_name}"

    task = run_nlp_training.delay(
        tenant_id=tenant.id,
        dataset_id=req.dataset_id,
        text_column=req.text_column,
        label_column=req.label_column,
        algorithm=req.algorithm,
        task_type=req.task_type,
        hyperparams=req.hyperparams,
        validation_config=req.validation.model_dump(),
        model_name=req.model_name or f"{algo_info['display_name']} — {dataset.name}",
        model_description=req.model_description or "",
        registry_name=registry_name,
    )

    return TrainResponse(
        message="Entrenamiento NLP lanzado correctamente.",
        task_id=task.id,
        status_url=f"/api/v1/nlp/training/status/{task.id}",
    )


# ──────────────────────────────────────────────────────────────────────────────
# GET /training/status/{task_id} — Polling de estado
# ──────────────────────────────────────────────────────────────────────────────


@router.get("/training/status/{task_id}")
def get_nlp_training_status(
    task_id: str,
    _user: User = Depends(require_viewer),
):
    """
    Consulta el estado de una tarea de entrenamiento NLP (Celery).
    """
    from celery.result import AsyncResult

    from app.worker.celery_app import celery_app

    result = AsyncResult(task_id, app=celery_app)

    response = {"task_id": task_id, "status": result.status}

    if result.ready():
        if result.successful():
            response["result"] = result.result
        else:
            response["error"] = (
                str(result.result) if result.result else "Unknown error"
            )

    return response
