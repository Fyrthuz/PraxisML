"""
Control Plane y Runtime HTTP para Model Serving Endpoints bajo demanda.
Permite publicar cualquier modelo registrado como un endpoint REST dedicado
autenticado mediante API Key (`X-API-Key` / `Bearer px_...`) o JWT.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.deps import (
    get_current_tenant,
    require_editor,
    require_viewer,
)
from app.core.config import settings
from app.core.rate_limit import limiter
from app.database import get_db
from app.models.endpoint_invocation_log import EndpointInvocationLog
from app.models.ml_model import MLModel
from app.models.model_endpoint import ModelEndpoint
from app.models.tenant import Tenant
from app.models.user import User
from app.platform.audit.service import record_audit_event
from app.platform.serving.service import invoke_model_endpoint

router = APIRouter()
invocation_router = APIRouter()


class EndpointCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=80)
    slug: Optional[str] = Field(default=None, max_length=64)
    description: Optional[str] = Field(default=None, max_length=300)
    model_id: str
    secondary_model_id: Optional[str] = None
    traffic_split_pct: int = Field(default=0, ge=0, le=100)
    workspace_id: Optional[str] = None
    preprocessing_pipeline_path: Optional[str] = None
    auth_mode: str = Field(default="api_key")


class EndpointUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=2, max_length=80)
    description: Optional[str] = Field(default=None, max_length=300)
    model_id: Optional[str] = None
    secondary_model_id: Optional[str] = None
    traffic_split_pct: Optional[int] = Field(default=None, ge=0, le=100)
    preprocessing_pipeline_path: Optional[str] = None
    status: Optional[str] = None


class EndpointInvokeRequest(BaseModel):
    inputs: Union[Dict[str, Any], List[Dict[str, Any]], List[str]]


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "model-endpoint"


def _serialize_endpoint(ep: ModelEndpoint, db: Session) -> Dict[str, Any]:
    primary = db.query(MLModel).filter(MLModel.id == ep.model_id).first()
    secondary = (
        db.query(MLModel).filter(MLModel.id == ep.secondary_model_id).first()
        if ep.secondary_model_id
        else None
    )
    meta = (primary.metrics_metadata or {}) if primary else {}
    return {
        "id": ep.id,
        "tenant_id": ep.tenant_id,
        "workspace_id": ep.workspace_id,
        "name": ep.name,
        "slug": ep.slug,
        "description": ep.description,
        "model_id": ep.model_id,
        "model_name": primary.name if primary else None,
        "model_version": primary.version if primary else None,
        "model_stage": primary.stage if primary else None,
        "feature_names": meta.get("feature_names") or [],
        "task_type": meta.get("task_type"),
        "framework": meta.get("framework"),
        "secondary_model_id": ep.secondary_model_id,
        "secondary_model_name": secondary.name if secondary else None,
        "traffic_split_pct": ep.traffic_split_pct,
        "preprocessing_pipeline_path": ep.preprocessing_pipeline_path,
        "status": ep.status,
        "auth_mode": ep.auth_mode,
        "invoke_path": f"/serving/v1/{ep.slug}/invocations",
        "api_invoke_path": f"{settings.API_V1_STR}/serving/endpoints/{ep.slug}/invocations",
        "request_count": ep.request_count,
        "error_count": ep.error_count,
        "avg_latency_ms": ep.avg_latency_ms,
        "last_invoked_at": ep.last_invoked_at.isoformat() if ep.last_invoked_at else None,
        "created_at": ep.created_at.isoformat() if ep.created_at else None,
        "updated_at": ep.updated_at.isoformat() if ep.updated_at else None,
    }


def _check_api_key_scope(request: Request, required_scope: str) -> None:
    if getattr(request.state, "auth_type", None) != "api_key":
        return
    scopes = getattr(request.state, "api_key_scopes", ["*"]) or ["*"]
    if "*" not in scopes and required_scope not in scopes:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"API key sin permiso requerido: '{required_scope}'.",
        )


@router.get("/endpoints")
def list_endpoints(
    _user: User = Depends(require_viewer),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    items = (
        db.query(ModelEndpoint)
        .filter(ModelEndpoint.tenant_id == tenant.id)
        .order_by(ModelEndpoint.created_at.desc())
        .all()
    )
    return {"total": len(items), "items": [_serialize_endpoint(ep, db) for ep in items]}


@router.post("/endpoints", status_code=201)
def create_endpoint(
    body: EndpointCreate,
    request: Request,
    user: User = Depends(require_editor),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    primary = (
        db.query(MLModel)
        .filter(MLModel.id == body.model_id, MLModel.tenant_id == tenant.id)
        .first()
    )
    if not primary:
        raise HTTPException(status_code=404, detail="Modelo principal no encontrado.")

    if body.secondary_model_id:
        secondary = (
            db.query(MLModel)
            .filter(MLModel.id == body.secondary_model_id, MLModel.tenant_id == tenant.id)
            .first()
        )
        if not secondary:
            raise HTTPException(status_code=404, detail="Modelo secundario (canary) no encontrado.")

    base_slug = _slugify(body.slug or body.name)
    slug = base_slug
    suffix = 2
    while db.query(ModelEndpoint).filter(ModelEndpoint.slug == slug).first():
        slug = f"{base_slug}-{suffix}"
        suffix += 1

    ep = ModelEndpoint(
        tenant_id=tenant.id,
        workspace_id=body.workspace_id,
        name=body.name,
        slug=slug,
        description=body.description,
        model_id=primary.id,
        secondary_model_id=body.secondary_model_id,
        traffic_split_pct=body.traffic_split_pct if body.secondary_model_id else 0,
        preprocessing_pipeline_path=(
            body.preprocessing_pipeline_path or primary.preprocessing_pipeline_path
        ),
        status="active",
        auth_mode=body.auth_mode,
        created_by=user.id,
    )
    db.add(ep)
    db.commit()
    db.refresh(ep)

    record_audit_event(
        db,
        tenant_id=tenant.id,
        workspace_id=ep.workspace_id,
        actor_id=user.id,
        action="endpoint.deployed",
        resource_type="model_endpoint",
        resource_id=ep.id,
        details={
            "slug": ep.slug,
            "model_id": ep.model_id,
            "secondary_model_id": ep.secondary_model_id,
            "traffic_split_pct": ep.traffic_split_pct,
        },
        request=request,
    )
    return _serialize_endpoint(ep, db)


@router.get("/endpoints/{endpoint_id}")
def get_endpoint(
    endpoint_id: str,
    _user: User = Depends(require_viewer),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    ep = (
        db.query(ModelEndpoint)
        .filter(
            (ModelEndpoint.id == endpoint_id) | (ModelEndpoint.slug == endpoint_id),
            ModelEndpoint.tenant_id == tenant.id,
        )
        .first()
    )
    if not ep:
        raise HTTPException(status_code=404, detail="Endpoint no encontrado.")
    return _serialize_endpoint(ep, db)


@router.patch("/endpoints/{endpoint_id}")
def update_endpoint(
    endpoint_id: str,
    body: EndpointUpdate,
    request: Request,
    user: User = Depends(require_editor),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    ep = (
        db.query(ModelEndpoint)
        .filter(
            (ModelEndpoint.id == endpoint_id) | (ModelEndpoint.slug == endpoint_id),
            ModelEndpoint.tenant_id == tenant.id,
        )
        .first()
    )
    if not ep:
        raise HTTPException(status_code=404, detail="Endpoint no encontrado.")

    if body.model_id is not None:
        primary = (
            db.query(MLModel)
            .filter(MLModel.id == body.model_id, MLModel.tenant_id == tenant.id)
            .first()
        )
        if not primary:
            raise HTTPException(status_code=404, detail="Modelo principal no encontrado.")
        ep.model_id = primary.id

    if body.secondary_model_id is not None:
        if body.secondary_model_id == "":
            ep.secondary_model_id = None
            ep.traffic_split_pct = 0
        else:
            secondary = (
                db.query(MLModel)
                .filter(MLModel.id == body.secondary_model_id, MLModel.tenant_id == tenant.id)
                .first()
            )
            if not secondary:
                raise HTTPException(status_code=404, detail="Modelo canary no encontrado.")
            ep.secondary_model_id = secondary.id

    if body.traffic_split_pct is not None:
        ep.traffic_split_pct = body.traffic_split_pct if ep.secondary_model_id else 0

    if body.status is not None:
        if body.status not in {"active", "paused"}:
            raise HTTPException(status_code=400, detail="Estado inválido. Usa 'active' o 'paused'.")
        ep.status = body.status

    if body.name is not None:
        ep.name = body.name
    if body.description is not None:
        ep.description = body.description
    if body.preprocessing_pipeline_path is not None:
        ep.preprocessing_pipeline_path = body.preprocessing_pipeline_path or None

    ep.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(ep)

    record_audit_event(
        db,
        tenant_id=tenant.id,
        workspace_id=ep.workspace_id,
        actor_id=user.id,
        action="endpoint.updated",
        resource_type="model_endpoint",
        resource_id=ep.id,
        details=body.model_dump(exclude_none=True),
        request=request,
    )
    return _serialize_endpoint(ep, db)


@router.delete("/endpoints/{endpoint_id}")
def delete_endpoint(
    endpoint_id: str,
    request: Request,
    user: User = Depends(require_editor),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    ep = (
        db.query(ModelEndpoint)
        .filter(
            (ModelEndpoint.id == endpoint_id) | (ModelEndpoint.slug == endpoint_id),
            ModelEndpoint.tenant_id == tenant.id,
        )
        .first()
    )
    if not ep:
        raise HTTPException(status_code=404, detail="Endpoint no encontrado.")

    slug = ep.slug
    ep_id = ep.id
    db.query(EndpointInvocationLog).filter(
        EndpointInvocationLog.endpoint_id == ep_id
    ).delete()
    db.delete(ep)
    db.commit()

    record_audit_event(
        db,
        tenant_id=tenant.id,
        actor_id=user.id,
        action="endpoint.deleted",
        resource_type="model_endpoint",
        resource_id=ep_id,
        details={"slug": slug},
        request=request,
    )
    return {"message": "Endpoint eliminado.", "id": ep_id, "slug": slug}


@router.get("/endpoints/{endpoint_id}/logs")
def get_endpoint_logs(
    endpoint_id: str,
    limit: int = Query(default=30, ge=1, le=200),
    _user: User = Depends(require_viewer),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """Lista el historial reciente de invocaciones del endpoint para depuración y drift."""
    ep = (
        db.query(ModelEndpoint)
        .filter(
            (ModelEndpoint.id == endpoint_id) | (ModelEndpoint.slug == endpoint_id),
            ModelEndpoint.tenant_id == tenant.id,
        )
        .first()
    )
    if not ep:
        raise HTTPException(status_code=404, detail="Endpoint no encontrado.")

    rows = (
        db.query(EndpointInvocationLog)
        .filter(
            EndpointInvocationLog.endpoint_id == ep.id,
            EndpointInvocationLog.tenant_id == tenant.id,
        )
        .order_by(EndpointInvocationLog.created_at.desc())
        .limit(limit)
        .all()
    )
    return {
        "endpoint_id": ep.id,
        "slug": ep.slug,
        "total": len(rows),
        "items": [
            {
                "id": r.id,
                "model_id": r.model_id,
                "variant": r.variant,
                "runtime": r.runtime,
                "status": r.status,
                "latency_ms": r.latency_ms,
                "input_count": r.input_count,
                "sample_input": r.sample_input,
                "sample_output": r.sample_output,
                "error_message": r.error_message,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ],
    }


def _handle_invocation(
    slug: str,
    body: EndpointInvokeRequest,
    request: Request,
    tenant: Tenant,
    db: Session,
) -> Dict[str, Any]:
    _check_api_key_scope(request, "serving:invoke")

    ep = (
        db.query(ModelEndpoint)
        .filter(ModelEndpoint.slug == slug, ModelEndpoint.tenant_id == tenant.id)
        .first()
    )
    if not ep:
        raise HTTPException(status_code=404, detail=f"Endpoint '{slug}' no encontrado.")
    if ep.status != "active":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Endpoint '{slug}' está en estado '{ep.status}'.",
        )

    try:
        return invoke_model_endpoint(ep, body.inputs, db)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error ejecutando predicción: {exc}")


@router.post("/endpoints/{slug}/invocations")
@limiter.limit(settings.RATE_LIMIT_INFERENCE)
def invoke_endpoint_v1(
    slug: str,
    body: EndpointInvokeRequest,
    request: Request,
    _user: User = Depends(require_viewer),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """Invoca un Model Endpoint dedicado mediante JWT o API Key (`X-API-Key`)."""
    return _handle_invocation(slug, body, request, tenant, db)


@invocation_router.post("/{slug}/invocations")
@limiter.limit(settings.RATE_LIMIT_INFERENCE)
def invoke_endpoint_serving(
    slug: str,
    body: EndpointInvokeRequest,
    request: Request,
    _user: User = Depends(require_viewer),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """Ruta corta de Serving Plane (`/serving/v1/{slug}/invocations`)."""
    return _handle_invocation(slug, body, request, tenant, db)
