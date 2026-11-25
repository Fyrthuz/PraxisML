"""
Servicio de auditoría transversal append-only sobre `audit_log` + Event Bus.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import Request
from sqlalchemy.orm import Session

from app.core.event_bus import get_event_bus
from app.models.audit_log import AuditLog

logger = logging.getLogger(__name__)


def record_audit_event(
    db: Session,
    *,
    tenant_id: str,
    action: str,
    resource_type: str,
    resource_id: Optional[str] = None,
    actor_id: Optional[str] = None,
    actor_type: str = "user",
    workspace_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    request: Optional[Request] = None,
    commit: bool = True,
) -> Optional[AuditLog]:
    """Registra una entrada inmutable en `audit_log` y publica un evento de dominio."""
    ip_address = None
    request_id = None
    if request is not None:
        if request.client:
            ip_address = request.client.host
        request_id = getattr(request.state, "request_id", None) or request.headers.get("x-request-id")
        if getattr(request.state, "auth_type", None) == "api_key" and actor_type == "user":
            actor_type = "api_key"

    entry = AuditLog(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        actor_id=actor_id,
        actor_type=actor_type,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        details=details or {},
        ip_address=ip_address,
        request_id=request_id,
    )
    try:
        db.add(entry)
        if commit:
            db.commit()
            db.refresh(entry)
    except Exception as exc:
        logger.warning("No se pudo persistir audit_log (%s): %s", action, exc)
        db.rollback()
        return None

    get_event_bus().publish(
        event_type=action,
        tenant_id=tenant_id,
        payload={
            "audit_id": entry.id,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "actor_id": actor_id,
            "actor_type": actor_type,
            **(details or {}),
        },
        source="audit",
    )
    return entry
