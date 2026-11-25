"""Gestión de API Keys / Personal Access Tokens (PATs) para acceso M2M y Model Serving."""

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.deps import get_current_tenant, require_editor, require_viewer
from app.database import get_db
from app.models.api_key import APIKey
from app.models.tenant import Tenant
from app.models.user import User
from app.platform.audit.service import record_audit_event

router = APIRouter()


class APIKeyCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=80)
    workspace_id: Optional[str] = None
    scopes: List[str] = Field(default_factory=lambda: ["serving:invoke", "models:read"])
    expires_in_days: Optional[int] = Field(default=90, ge=1, le=365)


def _serialize(item: APIKey, secret: Optional[str] = None) -> dict:
    payload = {
        "id": item.id,
        "name": item.name,
        "key_prefix": item.key_prefix,
        "workspace_id": item.workspace_id,
        "scopes": item.scopes or ["*"],
        "expires_at": item.expires_at.isoformat() if item.expires_at else None,
        "last_used_at": item.last_used_at.isoformat() if item.last_used_at else None,
        "revoked_at": item.revoked_at.isoformat() if item.revoked_at else None,
        "is_active": item.revoked_at is None,
        "created_at": item.created_at.isoformat() if item.created_at else None,
    }
    if secret is not None:
        payload["api_key"] = secret
    return payload


@router.get("/")
def list_api_keys(
    user: User = Depends(require_viewer),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    query = db.query(APIKey).filter(APIKey.tenant_id == tenant.id)
    if user.role != "admin":
        query = query.filter(APIKey.user_id == user.id)
    items = query.order_by(APIKey.created_at.desc()).all()
    return {"total": len(items), "items": [_serialize(k) for k in items]}


@router.post("/", status_code=201)
def create_api_key(
    body: APIKeyCreate,
    request: Request,
    user: User = Depends(require_editor),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """Genera una nueva API key (`px_...`). El secreto solo se muestra una vez."""
    raw_secret = f"px_{secrets.token_urlsafe(32)}"
    key_prefix = raw_secret[:12]
    key_hash = hashlib.sha256(raw_secret.encode("utf-8")).hexdigest()
    expires_at = (
        datetime.now(timezone.utc) + timedelta(days=body.expires_in_days)
        if body.expires_in_days
        else None
    )

    api_key = APIKey(
        tenant_id=tenant.id,
        workspace_id=body.workspace_id,
        user_id=user.id,
        name=body.name,
        key_prefix=key_prefix,
        key_hash_sha256=key_hash,
        scopes=body.scopes or ["*"],
        expires_at=expires_at,
    )
    db.add(api_key)
    db.commit()
    db.refresh(api_key)

    record_audit_event(
        db,
        tenant_id=tenant.id,
        workspace_id=body.workspace_id,
        actor_id=user.id,
        action="auth.key_created",
        resource_type="api_key",
        resource_id=api_key.id,
        details={"name": api_key.name, "prefix": key_prefix, "scopes": api_key.scopes},
        request=request,
    )
    return _serialize(api_key, secret=raw_secret)


@router.delete("/{key_id}")
def revoke_api_key(
    key_id: str,
    request: Request,
    user: User = Depends(require_editor),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    query = db.query(APIKey).filter(APIKey.id == key_id, APIKey.tenant_id == tenant.id)
    if user.role != "admin":
        query = query.filter(APIKey.user_id == user.id)
    api_key = query.first()
    if not api_key:
        raise HTTPException(status_code=404, detail="API key no encontrada.")

    api_key.revoked_at = datetime.now(timezone.utc)
    db.commit()

    record_audit_event(
        db,
        tenant_id=tenant.id,
        workspace_id=api_key.workspace_id,
        actor_id=user.id,
        action="auth.key_revoked",
        resource_type="api_key",
        resource_id=api_key.id,
        details={"name": api_key.name, "prefix": api_key.key_prefix},
        request=request,
    )
    return {"message": "API key revocada.", "id": api_key.id}
