import os
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import (
    get_current_tenant,
    require_admin,
    require_viewer,
)
from app.database import get_db
from app.models.tenant import Tenant
from app.models.user import User
from app.schemas.tenant import TenantCreate, TenantResponse, TenantUpdate

router = APIRouter()


@router.post("/", response_model=TenantResponse, status_code=status.HTTP_201_CREATED)
def create_tenant(
    tenant_in: TenantCreate,
    user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """
    Crea un nuevo tenant. Solo admins pueden crear tenants.
    """
    # 1. Crear registro en BD
    new_tenant = Tenant(name=tenant_in.name)
    db.add(new_tenant)
    db.commit()
    db.refresh(new_tenant)

    # 2. Actualizar el tenant_id del usuario creador (MVP: 1 usuario = 1 tenant)
    user.tenant_id = new_tenant.id
    db.commit()

    # 3. Sembrar alertas por defecto para el nuevo tenant
    from app.observability.models import AlertConfig
    from datetime import datetime, timezone

    default_alerts = [
        ("cost_spike", "inference_cost_usd", 2.0, "pct_change", 60),
        ("budget_near_limit", "cost_budget_remaining", 80.0, "lt", 60),
        ("budget_exceeded", "cost_budget_remaining", 0.0, "lt", 1440),
        ("latency_degradation", "inference_latency_ms", 2.0, "pct_change", 30),
        ("latency_spike", "inference_latency_ms", 5.0, "pct_change", 15),
        ("drift_warning", "drift_score", 0.15, "gt", 120),
        ("drift_critical", "drift_score", 0.3, "gt", 60),
        ("token_spike_llm", "inference_tokens_total", 3.0, "pct_change", 60),
        ("throughput_drop", "predictions_total", 0.5, "pct_change", 30),
        ("error_rate_spike", "errors_total", 0.05, "gt", 10),
        ("training_failed", "errors_total", 0.0, "gt", 60),
        ("quota_imminent", "quota_usage_pct", 90.0, "gt", 1440),
        ("model_zombie", "predictions_total", 0.0, "lt", 1440),
        ("gpu_underutilized", "training_duration_seconds", 20.0, "lt", 120),
    ]
    for alert_type, metric, threshold, comparison, cooldown in default_alerts:
        db.add(AlertConfig(
            tenant_id=new_tenant.id,
            alert_type=alert_type,
            metric_name=metric,
            threshold=threshold,
            comparison=comparison,
            cooldown_min=cooldown,
            channels=[{"type": "webhook", "url": ""}],
            enabled=True,
        ))
    db.commit()

    # 4. Crear carpeta local para almacenar sus archivos privados
    from app.core.config import settings
    tenant_dir = os.path.join(settings.DATA_DIR, "tenants", new_tenant.id, "datasets")
    os.makedirs(tenant_dir, exist_ok=True)

    return new_tenant


@router.get("/me", response_model=TenantResponse)
def get_my_active_tenant(
    tenant: Tenant = Depends(get_current_tenant),
    _user: User = Depends(require_viewer),
    db: Session = Depends(get_db),
):
    """Retorna el tenant actual del usuario autenticado (basado en token)."""
    return tenant


@router.get("/my_tenants", response_model=List[TenantResponse])
def get_my_tenants(
    user: User = Depends(require_viewer),
    db: Session = Depends(get_db),
):
    """Lista todos los tenants a los que pertenece el usuario (MVP: solo 1 activo)."""
    tenant = db.query(Tenant).filter(Tenant.id == user.tenant_id).first()
    return [tenant] if tenant else []


@router.patch("/{tenant_id}", response_model=TenantResponse)
def update_tenant_quotas(
    tenant_id: str,
    update_in: TenantUpdate,
    user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """
    Actualiza las cuotas de un tenant. Solo admins del tenant pueden hacerlo.
    """
    tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
    if not tenant:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tenant no encontrado")

    # Verificar que el admin pertenece a este tenant
    if user.tenant_id != tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="No tienes permisos para modificar este tenant.",
        )

    update_data = update_in.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(tenant, field, value)

    db.commit()
    db.refresh(tenant)
    return tenant


@router.get("/{tenant_id}", response_model=TenantResponse)
def get_tenant(
    tenant_id: str,
    user: User = Depends(require_viewer),
    db: Session = Depends(get_db),
):
    tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
    if not tenant:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tenant no encontrado")
    return tenant
