"""
API route para inferencia NLP.

Endpoints:
  POST /api/v1/nlp/predict            → Clasificar texto / extraer entidades (síncrono)
  POST /api/v1/nlp/predict/batch      → Batch inference sobre CSV (Celery, 202)
  GET  /api/v1/nlp/predict/batch/{prediction_id}/result → Resultados batch JSON
  WS   /api/v1/nlp/streaming/{model_id} → Clasificación en tiempo real
"""

import json
import logging
import os
from typing import Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    WebSocket,
    status,
)
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import (
    check_prediction_quota,
    get_current_tenant,
    require_editor,
    require_viewer,
)
from app.core.config import settings
from app.core.rate_limit import limiter
from app.database import get_db
from app.models.ml_model import MLModel
from app.models.tenant import Tenant
from app.models.user import User

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Schemas ──────────────────────────────────────────────────────────────────


class PredictRequest(BaseModel):
    model_id: str
    text: str
    explain: bool = False


class PredictResponse(BaseModel):
    label: Optional[str] = None
    confidence: Optional[float] = None
    entities: list = []
    tokens: list = []
    latency_ms: float = 0.0
    explanation: Optional[dict] = None


_NLP_TASK_TYPES = {"text_classification", "token_classification", "sentiment_analysis", "ner"}


def _get_nlp_model(db: Session, model_id: str, tenant_id: str) -> MLModel:
    ml_model = (
        db.query(MLModel)
        .filter(MLModel.id == model_id, MLModel.tenant_id == tenant_id)
        .first()
    )
    if not ml_model:
        raise HTTPException(status_code=404, detail="Modelo no encontrado.")
    metadata = ml_model.metrics_metadata or {}
    domain = str(metadata.get("domain") or "").lower()
    framework = str(metadata.get("framework") or "").lower()
    task_type = str(metadata.get("task_type") or "").lower()
    is_nlp = (
        domain == "nlp"
        or framework == "huggingface"
        or task_type in _NLP_TASK_TYPES
        or bool(metadata.get("onnx_path") or metadata.get("tokenizer_path"))
    )
    if not is_nlp:
        raise HTTPException(
            status_code=400,
            detail="El modelo seleccionado no pertenece al dominio NLP. Usa inferencia tabular para este modelo.",
        )
    return ml_model


# ──────────────────────────────────────────────────────────────────────────────
# POST /predict — Inferencia síncrona de un texto
# ──────────────────────────────────────────────────────────────────────────────


@router.post("/predict", response_model=PredictResponse)
@limiter.limit(settings.RATE_LIMIT_NLP_INFERENCE)
def predict_nlp(
    request: Request,
    req: PredictRequest,
    _user: User = Depends(require_editor),
    tenant: Tenant = Depends(check_prediction_quota),
    db: Session = Depends(get_db),
):
    """
    Clasifica o extrae entidades de un texto con un modelo NLP del tenant.

    Requiere rol **editor** o superior. Rate limited a {RATE_LIMIT_NLP_INFERENCE}.
    """
    ml_model = _get_nlp_model(db, req.model_id, tenant.id)
    metadata = ml_model.metrics_metadata or {}

    from app.domains.nlp.inference import get_nlp_inference_service

    service = get_nlp_inference_service()
    handle = service.load_model(ml_model)
    result = service.predict(handle, req.text)

    if req.explain:
        try:
            from app.core_ml.explainability import explain_nlp_text

            result["explanation"] = explain_nlp_text(service, handle, req.text, result)
        except Exception as exc:
            logger.warning("NLP explainability skipped: %s", exc)

    try:
        tokens_in = len(result.get("tokens", []))
        service.record_usage(
            db=db,
            tenant_id=tenant.id,
            model_id=ml_model.id,
            tokens_in=tokens_in,
            latency_ms=result.get("latency_ms", 0),
            tags={
                "algorithm": metadata.get("algorithm", ""),
                "task_type": metadata.get("task_type", "text_classification"),
            },
        )
        db.commit()
    except Exception as exc:
        logger.warning("Usage recording failed (non-critical): %s", exc)

    return PredictResponse(**result)


# ──────────────────────────────────────────────────────────────────────────────
# POST /predict/batch — Batch inference sobre CSV (asíncrono)
# ──────────────────────────────────────────────────────────────────────────────


@router.post("/predict/batch", status_code=status.HTTP_202_ACCEPTED)
@limiter.limit(settings.RATE_LIMIT_NLP_INFERENCE)
def predict_nlp_batch(
    request: Request,
    model_id: str = Form(...),
    text_column: str = Form("text"),
    file: UploadFile = File(...),
    _user: User = Depends(require_editor),
    tenant: Tenant = Depends(check_prediction_quota),
    db: Session = Depends(get_db),
):
    """Encola inferencia batch NLP sobre un CSV. Devuelve prediction_id para polling."""
    from app.models.prediction import Prediction

    _get_nlp_model(db, model_id, tenant.id)

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400, detail="El batch NLP requiere un archivo .csv"
        )

    tenant_tmp_dir = os.path.join(settings.DATA_DIR, "tenants", tenant.id, "tmp")
    os.makedirs(tenant_tmp_dir, exist_ok=True)
    file_path = os.path.join(tenant_tmp_dir, f"nlp_{file.filename}")

    with open(file_path, "wb") as buffer:
        import shutil

        shutil.copyfileobj(file.file, buffer)

    prediction = Prediction(
        status="PENDING",
        method="nlp_batch",
        dataset_id=None,
        model_id=model_id,
        tenant_id=tenant.id,
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)

    from app.worker.tasks.predict_nlp import run_nlp_batch_inference

    task = run_nlp_batch_inference.delay(
        tenant_id=tenant.id,
        model_id=model_id,
        prediction_id=prediction.id,
        input_file_path=file_path,
        text_column=text_column,
    )
    prediction.task_id = task.id
    db.commit()

    return {
        "message": "Batch NLP encolado exitosamente.",
        "prediction_id": prediction.id,
        "task_id": task.id,
        "status_url": f"/api/v1/tabular/predictions/status/{task.id}",
        "result_url": f"/api/v1/nlp/predict/batch/{prediction.id}/result",
    }


# ──────────────────────────────────────────────────────────────────────────────
# GET /predict/batch/{prediction_id}/result — Resultados batch
# ──────────────────────────────────────────────────────────────────────────────


@router.get("/predict/batch/{prediction_id}/result")
def get_nlp_batch_result(
    prediction_id: str,
    _user: User = Depends(require_viewer),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """Devuelve los resultados JSON de una inferencia batch NLP."""
    from app.models.prediction import Prediction

    prediction = (
        db.query(Prediction)
        .filter(Prediction.id == prediction_id, Prediction.tenant_id == tenant.id)
        .first()
    )
    if not prediction:
        raise HTTPException(status_code=404, detail="Predicción no encontrada.")
    if prediction.status != "COMPLETED" or not prediction.result_path:
        return {"status": prediction.status, "result": None}

    path = prediction.result_path
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Resultados no encontrados.")

    with open(path, "r", encoding="utf-8") as f:
        return {"status": "COMPLETED", "result": json.load(f)}


# ──────────────────────────────────────────────────────────────────────────────
# WS /streaming/{model_id} — Clasificación en tiempo real
# ──────────────────────────────────────────────────────────────────────────────


@router.websocket("/streaming/{model_id}")
async def websocket_nlp_predict(
    websocket: WebSocket,
    model_id: str,
    token: str = Query(..., description="JWT token"),
):
    """
    WebSocket para inferencia NLP en tiempo real.
    Recibe {text} → envía {label, confidence, entities, tokens, latency_ms}.
    Autenticación JWT via query param: ?token=<jwt>
    """
    from fastapi import WebSocketDisconnect

    from app.core.security import decode_token
    from app.database import SessionLocal

    await websocket.accept()

    try:
        payload = decode_token(token)
        if not payload:
            await websocket.close(code=1008, reason="Invalid token")
            return
    except Exception:
        await websocket.close(code=1008, reason="Token decoding error")
        return

    tenant_id = payload.get("tenant_id")
    user_id = payload.get("sub")
    if not tenant_id and user_id:
        db = SessionLocal()
        try:
            from app.models.user import User

            user = db.query(User).filter(User.id == user_id).first()
            if user:
                tenant_id = user.tenant_id
        except Exception:
            pass
        finally:
            db.close()

    if not tenant_id:
        await websocket.close(code=1008, reason="Tenant not found")
        return

    db = SessionLocal()
    try:
        ml_model = (
            db.query(MLModel)
            .filter(MLModel.id == model_id, MLModel.tenant_id == tenant_id)
            .first()
        )
        if not ml_model:
            await websocket.close(code=1008, reason="Modelo no encontrado")
            return

        from app.domains.nlp.inference import get_nlp_inference_service

        service = get_nlp_inference_service()
        handle = service.load_model(ml_model)

        while True:
            data = await websocket.receive_text()
            try:
                row = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "JSON inválido"})
                continue
            text = row.get("text", "")
            if not text:
                await websocket.send_json({"error": "Campo 'text' requerido"})
                continue
            result = service.predict(handle, text)
            await websocket.send_json(result)

    except WebSocketDisconnect:
        logger.info("NLP WebSocket desconectado: %s", model_id)
    except Exception as exc:
        logger.error("Error en WebSocket NLP: %s", exc)
        try:
            await websocket.close(code=1011, reason=str(exc))
        except Exception:
            pass
    finally:
        db.close()
