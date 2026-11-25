"""
Motor de invocación para Model Serving Endpoints bajo demanda.
Delega la ejecución al `ModelRuntimeRegistry` (agnóstico al framework),
gestiona enrutamiento Canary/A-B, registra logs ligeros de invocación,
actualiza métricas de latencia y publica eventos de dominio.
"""

from __future__ import annotations

import logging
import random
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, Union

from sqlalchemy.orm import Session

from app.core.event_bus import get_event_bus
from app.models.endpoint_invocation_log import EndpointInvocationLog
from app.models.ml_model import MLModel
from app.models.model_endpoint import ModelEndpoint
from app.platform.serving.runtimes import get_model_runtime_registry

logger = logging.getLogger(__name__)


def _select_target_model(endpoint: ModelEndpoint, db: Session) -> Tuple[MLModel, str]:
    """Selecciona modelo primario o canary según traffic_split_pct."""
    use_canary = (
        bool(endpoint.secondary_model_id)
        and endpoint.traffic_split_pct > 0
        and random.randint(1, 100) <= endpoint.traffic_split_pct
    )
    target_id = endpoint.secondary_model_id if use_canary else endpoint.model_id
    variant = "canary" if use_canary else "primary"

    ml_model = (
        db.query(MLModel)
        .filter(MLModel.id == target_id, MLModel.tenant_id == endpoint.tenant_id)
        .first()
    )
    if ml_model is None:
        raise ValueError(
            f"Modelo '{target_id}' no encontrado para el endpoint '{endpoint.slug}'."
        )
    return ml_model, variant


def _sample_payload(value: Any, max_items: int = 3) -> Any:
    if isinstance(value, list):
        return value[:max_items]
    return value


def invoke_model_endpoint(
    endpoint: ModelEndpoint,
    inputs: Union[Dict[str, Any], List[Dict[str, Any]], List[str]],
    db: Session,
) -> Dict[str, Any]:
    """Ejecuta una invocación síncrona contra un ModelEndpoint y persiste métricas y log."""
    start = time.perf_counter()
    target_model, variant = _select_target_model(endpoint, db)
    runtime = get_model_runtime_registry().resolve(target_model)

    try:
        pipeline_path = (
            endpoint.preprocessing_pipeline_path
            or target_model.preprocessing_pipeline_path
        )
        output = runtime.predict(
            ml_model=target_model,
            inputs=inputs,
            pipeline_path=pipeline_path,
        )
        count = output["count"]

        latency_ms = round((time.perf_counter() - start) * 1000.0, 2)
        prev_count = endpoint.request_count or 0
        new_count = prev_count + 1
        prev_avg = endpoint.avg_latency_ms or 0.0
        endpoint.request_count = new_count
        endpoint.avg_latency_ms = round(((prev_avg * prev_count) + latency_ms) / new_count, 2)
        endpoint.last_invoked_at = datetime.now(timezone.utc)

        log_row = EndpointInvocationLog(
            endpoint_id=endpoint.id,
            tenant_id=endpoint.tenant_id,
            model_id=target_model.id,
            variant=variant,
            runtime=runtime.runtime_name,
            status="ok",
            latency_ms=latency_ms,
            input_count=count,
            sample_input=_sample_payload(inputs),
            sample_output={
                "predictions": _sample_payload(output["predictions"]),
                "probabilities": _sample_payload(output["probabilities"])
                if output["probabilities"] is not None
                else None,
            },
        )
        db.add(log_row)
        db.commit()

        get_event_bus().publish(
            event_type="endpoint.invoked",
            tenant_id=endpoint.tenant_id,
            payload={
                "endpoint_id": endpoint.id,
                "slug": endpoint.slug,
                "model_id": target_model.id,
                "runtime": runtime.runtime_name,
                "variant": variant,
                "count": count,
                "latency_ms": latency_ms,
            },
            source="serving",
        )

        return {
            "endpoint_id": endpoint.id,
            "endpoint_slug": endpoint.slug,
            "model_id": target_model.id,
            "model_name": target_model.name,
            "model_version": target_model.version,
            "runtime": runtime.runtime_name,
            "variant": variant,
            "latency_ms": latency_ms,
            "count": count,
            "predictions": output["predictions"],
            "probabilities": output["probabilities"],
        }
    except Exception as exc:
        latency_ms = round((time.perf_counter() - start) * 1000.0, 2)
        endpoint.error_count = (endpoint.error_count or 0) + 1
        endpoint.last_invoked_at = datetime.now(timezone.utc)
        try:
            db.add(
                EndpointInvocationLog(
                    endpoint_id=endpoint.id,
                    tenant_id=endpoint.tenant_id,
                    model_id=target_model.id,
                    variant=variant,
                    runtime=runtime.runtime_name,
                    status="error",
                    latency_ms=latency_ms,
                    input_count=len(inputs) if isinstance(inputs, list) else 1,
                    sample_input=_sample_payload(inputs),
                    error_message=str(exc)[:400],
                )
            )
            db.commit()
        except Exception:
            db.rollback()
        raise
