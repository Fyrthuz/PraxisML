from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    status,
    Form,
    File,
    UploadFile,
    Query,
)
from sqlalchemy.orm import Session
from typing import List, Optional

from app.database import get_db
from app.models.prediction import Prediction
from app.models.dataset import Dataset
from app.models.ml_model import MLModel
from app.models.user import User
from app.models.tenant import Tenant
from app.schemas.prediction import PredictionResponse
from app.schemas.pagination import PaginatedResponse
from app.api.deps import (
    get_current_tenant,
    require_editor,
    require_viewer,
    check_prediction_quota,
)
from app.core.rate_limit import limiter
from app.core.config import settings
from pydantic import BaseModel
import os
import shutil
import numpy as np
import io
from fastapi.responses import StreamingResponse
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


class PredictionRequest(BaseModel):
    dataset_id: str
    model_id: str
    uncertainty_method: str = "mc_dropout"


@router.post("/predict", status_code=status.HTTP_202_ACCEPTED)
@limiter.limit(settings.RATE_LIMIT_INFERENCE)
def request_prediction(
    request: Request,
    req: PredictionRequest,
    _user: User = Depends(require_editor),
    tenant: Tenant = Depends(check_prediction_quota),
    db: Session = Depends(get_db),
):
    """
    Endpoint que no bloquea la petición web.
    Crea el registro Prediction en BD, lo encola en Celery y devuelve el ID para polling.

    Requiere rol **editor** o superior. Rate limited a {RATE_LIMIT_INFERENCE}.
    Valida cuota diaria de predicciones del tenant.
    """
    from app.worker.tasks.predict import run_heavy_inference

    # Validar existencia de dataset y modelo para fallo rápido
    dataset = db.query(Dataset).filter(Dataset.id == req.dataset_id).first()
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Dataset no encontrado."
        )

    ml_model = db.query(MLModel).filter(MLModel.id == req.model_id).first()
    if not ml_model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Modelo no encontrado."
        )

    # Crear registro PENDING en BD antes de encolar
    prediction = Prediction(
        status="PENDING",
        method=req.uncertainty_method,
        dataset_id=req.dataset_id,
        model_id=req.model_id,
        tenant_id=tenant.id,
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)

    # Enviar tarea a Redis/Celery
    task = run_heavy_inference.delay(
        tenant_id=tenant.id,
        dataset_id=req.dataset_id,
        model_id=req.model_id,
        method=req.uncertainty_method,
        prediction_id=prediction.id,
    )

    # Guardar el task_id de Celery en el registro
    prediction.task_id = task.id
    db.commit()

    return {
        "message": "Predicción encolada exitosamente.",
        "prediction_id": prediction.id,
        "task_id": task.id,
        "status_url": f"/api/v1/predictions/status/{task.id}",
        "result_url": f"/api/v1/predictions/{prediction.id}",
    }


@router.post("/predictions/predict/single", status_code=status.HTTP_202_ACCEPTED)
@limiter.limit(settings.RATE_LIMIT_INFERENCE)
def request_single_prediction(
    request: Request,
    model_id: str = Form(...),
    uncertainty_method: str = Form("none"),
    features: str = Form(...),
    explain: bool = Form(False),
    _user: User = Depends(require_editor),
    tenant: Tenant = Depends(check_prediction_quota),
    db: Session = Depends(get_db),
):
    """
    Endpoint para inferencia de una sola muestra tabular vía JSON de características.

    Requiere rol **editor** o superior. Rate limited a {RATE_LIMIT_INFERENCE}.
    """
    from app.worker.tasks.single_predict import run_single_tabular_inference
    import json

    try:
        features_dict = json.loads(features)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Las características proporcionadas no son un JSON válido.",
        )

    ml_model = db.query(MLModel).filter(MLModel.id == model_id).first()
    if not ml_model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Modelo no encontrado."
        )

    # Crear registro Prediction
    prediction = Prediction(
        status="PENDING",
        method=uncertainty_method,
        dataset_id=None,
        model_id=model_id,
        tenant_id=tenant.id,
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)

    # Enviar tarea a Celery
    task = run_single_tabular_inference.delay(
        tenant_id=tenant.id,
        image_path=None,
        model_id=model_id,
        method=uncertainty_method,
        prediction_id=prediction.id,
        features=features_dict,
        explain=explain,
    )

    prediction.task_id = task.id
    db.commit()

    return {
        "message": "Predicción tabular individual encolada.",
        "prediction_id": prediction.id,
        "task_id": task.id,
    }


@router.post("/predictions/predict/batch", status_code=status.HTTP_202_ACCEPTED)
@limiter.limit(settings.RATE_LIMIT_INFERENCE)
def request_batch_prediction(
    request: Request,
    model_id: str = Form(...),
    uncertainty_method: str = Form("none"),
    file: UploadFile = File(...),
    _user: User = Depends(require_editor),
    tenant: Tenant = Depends(check_prediction_quota),
    db: Session = Depends(get_db),
):
    """
    Endpoint para inferencia en batch subiendo directamente un archivo CSV/Excel.

    Requiere rol **editor** o superior. Rate limited a {RATE_LIMIT_INFERENCE}.
    """
    from app.worker.tasks.predict import run_heavy_inference

    ml_model = db.query(MLModel).filter(MLModel.id == model_id).first()
    if not ml_model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Modelo no encontrado."
        )

    # Guardar archivo temporalmente
    tenant_tmp_dir = os.path.join(settings.DATA_DIR, "tenants", tenant.id, "tmp")
    os.makedirs(tenant_tmp_dir, exist_ok=True)
    file_path = os.path.join(tenant_tmp_dir, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Crear registro Prediction
    prediction = Prediction(
        status="PENDING",
        method=uncertainty_method,
        dataset_id=None,  # No hay un dataset_id de la tabla Dataset
        model_id=model_id,
        tenant_id=tenant.id,
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)

    # Enviar tarea a Celery. Pasamos el file_path directamente.
    # Necesitamos que run_heavy_inference acepte dataset_id=None si hay un input_file_path.
    task = run_heavy_inference.delay(
        tenant_id=tenant.id,
        dataset_id=None,
        model_id=model_id,
        method=uncertainty_method,
        prediction_id=prediction.id,
        input_file_path=file_path,  # Nuevo argumento para la tarea
    )

    prediction.task_id = task.id
    db.commit()

    return {
        "message": "Predicción en batch encolada exitosamente.",
        "prediction_id": prediction.id,
        "task_id": task.id,
    }


@router.get("/predictions/status/{task_id}")
def get_prediction_status(
    task_id: str,
    _user: User = Depends(require_viewer),
):
    """
    Polling del estado en tiempo real vía Celery AsyncResult.
    Requiere rol **viewer** o superior.
    """
    from celery.result import AsyncResult
    from app.worker.celery_app import celery_app

    task_result = AsyncResult(task_id, app=celery_app)

    response = {
        "task_id": task_id,
        "status": task_result.status,
    }

    if task_result.state == "SUCCESS":
        response["result"] = task_result.result
    elif task_result.state == "FAILURE":
        response["error"] = str(task_result.info)
    elif task_result.state == "PROGRESS":
        response["progress"] = task_result.info.get("status", "")

    return response


@router.get("/predictions", response_model=PaginatedResponse[PredictionResponse])
def list_predictions(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    _user: User = Depends(require_viewer),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """
    Obtiene las predicciones del tenant con paginación.
    Requiere rol **viewer** o superior.
    """
    query = (
        db.query(Prediction)
        .filter(Prediction.tenant_id == tenant.id)
        .order_by(Prediction.created_at.desc())
    )

    total = query.count()
    total_pages = (total + page_size - 1) // page_size if total > 0 else 0

    predictions = query.offset((page - 1) * page_size).limit(page_size).all()

    return PaginatedResponse(
        items=predictions,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
        has_next=page < total_pages,
        has_prev=page > 1,
    )


@router.get("/predictions/{prediction_id}", response_model=PredictionResponse)
def get_prediction_result(
    prediction_id: str,
    _user: User = Depends(require_viewer),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """
    Obtiene el resultado completo de una predicción almacenada en BD.
    Requiere rol **viewer** o superior.
    """
    prediction = (
        db.query(Prediction)
        .filter(Prediction.id == prediction_id, Prediction.tenant_id == tenant.id)
        .first()
    )
    if not prediction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Predicción {prediction_id} no encontrada.",
        )
    return prediction


@router.get("/predictions/{prediction_id}/data")
def get_prediction_data(
    prediction_id: str,
    _user: User = Depends(require_viewer),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """
    Retorna el contenido crudo de los ficheros de predicción e incertidumbre (.npy) como JSON.
    Requiere rol **viewer** o superior.
    """
    prediction = (
        db.query(Prediction)
        .filter(Prediction.id == prediction_id, Prediction.tenant_id == tenant.id)
        .first()
    )

    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")

    results = {}

    try:
        if prediction.result_path and os.path.exists(prediction.result_path):
            arr = np.load(prediction.result_path)
            # Replace NaNs and Infinities with None for JSON compliance
            cleaned = np.where(np.logical_or(np.isnan(arr), np.isinf(arr)), None, arr)
            results["prediction"] = cleaned.tolist()

        if prediction.uncertainty_path and os.path.exists(prediction.uncertainty_path):
            arr = np.load(prediction.uncertainty_path)
            cleaned = np.where(np.logical_or(np.isnan(arr), np.isinf(arr)), None, arr)
            results["uncertainty"] = cleaned.tolist()

        if (
            prediction.input_image_path
            and os.path.exists(prediction.input_image_path)
            and prediction.input_image_path.endswith(".npy")
        ):
            arr = np.load(prediction.input_image_path)
            # For input data, we also sanitize just in case
            if arr.dtype.kind in "fc":
                cleaned = np.where(
                    np.logical_or(np.isnan(arr), np.isinf(arr)), None, arr
                )
                results["input_data"] = cleaned.tolist()
            else:
                results["input_data"] = arr.tolist()

    except Exception as e:
        logger.error(f"Error loading prediction data for {prediction_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error reading result files: {str(e)}"
        )

    return results


@router.get("/predictions/{prediction_id}/visualization/{img_type}")
def get_prediction_visualization(
    prediction_id: str,
    img_type: str,
    _user: User = Depends(require_viewer),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """
    Convierte el resultado (.npy) al vuelo en un archivo PNG visualizable por el frontend.
    img_type: 'result' o 'uncertainty'
    Requiere rol **viewer** o superior.
    """
    prediction = (
        db.query(Prediction)
        .filter(Prediction.id == prediction_id, Prediction.tenant_id == tenant.id)
        .first()
    )

    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")

    # Serve the original uploaded image directly
    if img_type == "original":
        original_path = prediction.input_image_path
        if not original_path or not os.path.exists(original_path):
            raise HTTPException(status_code=404, detail="Original image not found")

        # Determine media type from extension
        ext = os.path.splitext(original_path)[1].lower()
        media_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
        }
        media_type = media_types.get(ext, "image/png")

        return StreamingResponse(open(original_path, "rb"), media_type=media_type)

    path_to_load = (
        prediction.result_path if img_type == "result" else prediction.uncertainty_path
    )
    if not path_to_load or not os.path.exists(path_to_load):
        raise HTTPException(
            status_code=404, detail=f"{img_type} file not found or not generated yet"
        )

    try:
        arr = np.load(path_to_load)
        # Handle shape like (1, 1, H, W) or (B, C, H, W)
        arr = np.squeeze(arr)

        # In case of multiple masks (e.g. multi-class), we take the argmax or just the first channel for now
        # Assuming typical (H, W) or (C, H, W). If (C, H, W), take first channel.
        if arr.ndim == 3:
            arr = arr[0]

        # Normalize to 0-255 uint8
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
        else:
            arr = np.zeros_like(arr)

        arr = arr.astype(np.uint8)

        # Apply pseudo-color for uncertainty map if desired, or just grayscale via PIL
        from PIL import Image

        img = Image.fromarray(arr).convert("L")

        # Si es uncertainty mapeamos a colormap viridis o similar (Opcional, pero para UI rápido sirve RGB)
        if img_type == "uncertainty":
            import matplotlib.pyplot as plt

            cmap = plt.get_cmap(
                "inferno"
            )  # Usando heatmap color map para error/uncertainty
            # normalize array between 0-1 for cmap
            arr_norm = arr.astype(np.float32) / 255.0
            colored_img = (cmap(arr_norm)[:, :, :3] * 255).astype(np.uint8)
            img = Image.fromarray(colored_img)

        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)

        return StreamingResponse(img_byte_arr, media_type="image/png")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating visualization: {str(e)}"
        )
