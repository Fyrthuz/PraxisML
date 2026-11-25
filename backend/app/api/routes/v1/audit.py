"""Endpoints de auditoría transversal y eventos de dominio."""

from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import get_current_tenant, require_viewer
from app.core.event_bus import get_event_bus
from app.database import get_db
from app.models.audit_log import AuditLog
from app.models.tenant import Tenant
from app.models.user import User

router = APIRouter()


@router.get("/")
def list_audit_logs(
    action: Optional[str] = Query(default=None),
    resource_type: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    _user: User = Depends(require_viewer),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """Lista las últimas entradas del audit log del tenant actual."""
    query = db.query(AuditLog).filter(AuditLog.tenant_id == tenant.id)
    if action:
        query = query.filter(AuditLog.action == action)
    if resource_type:
        query = query.filter(AuditLog.resource_type == resource_type)

    rows = query.order_by(AuditLog.created_at.desc()).limit(limit).all()
    return {
        "total": len(rows),
        "items": [
            {
                "id": r.id,
                "tenant_id": r.tenant_id,
                "workspace_id": r.workspace_id,
                "actor_id": r.actor_id,
                "actor_type": r.actor_type,
                "action": r.action,
                "resource_type": r.resource_type,
                "resource_id": r.resource_id,
                "details": r.details or {},
                "ip_address": r.ip_address,
                "request_id": r.request_id,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ],
    }


@router.get("/events")
def list_recent_domain_events(
    limit: int = Query(default=50, ge=1, le=200),
    _user: User = Depends(require_viewer),
    tenant: Tenant = Depends(get_current_tenant),
):
    """Devuelve los últimos eventos publicados en Redis Streams para el tenant."""
    events = get_event_bus().recent_events(limit=limit, tenant_id=tenant.id)
    return {"total": len(events), "events": events}
