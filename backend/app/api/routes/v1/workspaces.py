"""Gestión de Workspaces (Tenant -> Workspace -> Environment)."""

import re
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.deps import get_current_tenant, require_admin, require_editor, require_viewer
from app.database import get_db
from app.models.tenant import Tenant
from app.models.user import User
from app.models.workspace import Workspace
from app.platform.audit.service import record_audit_event

router = APIRouter()

VALID_ENVIRONMENTS = {"dev", "staging", "prod"}
VALID_COMPUTE_PROFILES = {"cpu-light", "cpu-heavy", "gpu-train"}


class WorkspaceCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=80)
    slug: Optional[str] = Field(default=None, max_length=64)
    environment: str = Field(default="dev")
    default_compute_profile: str = Field(default="cpu-light")
    budget_limit_usd: Optional[float] = Field(default=None, ge=0)


class WorkspaceUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=2, max_length=80)
    environment: Optional[str] = None
    default_compute_profile: Optional[str] = None
    budget_limit_usd: Optional[float] = Field(default=None, ge=0)
    is_active: Optional[bool] = None


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "workspace"


def _serialize(ws: Workspace) -> dict:
    return {
        "id": ws.id,
        "tenant_id": ws.tenant_id,
        "name": ws.name,
        "slug": ws.slug,
        "environment": ws.environment,
        "default_compute_profile": ws.default_compute_profile,
        "budget_limit_usd": ws.budget_limit_usd,
        "is_active": ws.is_active,
        "created_at": ws.created_at.isoformat() if ws.created_at else None,
    }


def _ensure_default_workspace(db: Session, tenant_id: str) -> list[Workspace]:
    items = (
        db.query(Workspace)
        .filter(Workspace.tenant_id == tenant_id)
        .order_by(Workspace.created_at.asc())
        .all()
    )
    if not items:
        default_ws = Workspace(
            tenant_id=tenant_id,
            name="Default Workspace",
            slug="default",
            environment="dev",
            default_compute_profile="cpu-light",
        )
        db.add(default_ws)
        db.commit()
        db.refresh(default_ws)
        items = [default_ws]
    return items


@router.get("/")
def list_workspaces(
    _user: User = Depends(require_viewer),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    items = _ensure_default_workspace(db, tenant.id)
    return {"total": len(items), "items": [_serialize(ws) for ws in items]}


@router.post("/", status_code=201)
def create_workspace(
    body: WorkspaceCreate,
    request: Request,
    user: User = Depends(require_editor),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    if body.environment not in VALID_ENVIRONMENTS:
        raise HTTPException(status_code=400, detail=f"Entorno inválido. Usa: {sorted(VALID_ENVIRONMENTS)}")
    if body.default_compute_profile not in VALID_COMPUTE_PROFILES:
        raise HTTPException(
            status_code=400,
            detail=f"Compute profile inválido. Usa: {sorted(VALID_COMPUTE_PROFILES)}",
        )

    slug = _slugify(body.slug or body.name)
    exists = (
        db.query(Workspace)
        .filter(Workspace.tenant_id == tenant.id, Workspace.slug == slug)
        .first()
    )
    if exists:
        raise HTTPException(status_code=409, detail=f"Ya existe un workspace con slug '{slug}'.")

    ws = Workspace(
        tenant_id=tenant.id,
        name=body.name,
        slug=slug,
        environment=body.environment,
        default_compute_profile=body.default_compute_profile,
        budget_limit_usd=body.budget_limit_usd,
    )
    db.add(ws)
    db.commit()
    db.refresh(ws)

    record_audit_event(
        db,
        tenant_id=tenant.id,
        workspace_id=ws.id,
        actor_id=user.id,
        action="workspace.created",
        resource_type="workspace",
        resource_id=ws.id,
        details={"slug": ws.slug, "environment": ws.environment},
        request=request,
    )
    return _serialize(ws)


@router.patch("/{workspace_id}")
def update_workspace(
    workspace_id: str,
    body: WorkspaceUpdate,
    request: Request,
    user: User = Depends(require_editor),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    ws = (
        db.query(Workspace)
        .filter(Workspace.id == workspace_id, Workspace.tenant_id == tenant.id)
        .first()
    )
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace no encontrado.")

    if body.environment is not None:
        if body.environment not in VALID_ENVIRONMENTS:
            raise HTTPException(status_code=400, detail="Entorno inválido.")
        ws.environment = body.environment
    if body.default_compute_profile is not None:
        if body.default_compute_profile not in VALID_COMPUTE_PROFILES:
            raise HTTPException(status_code=400, detail="Compute profile inválido.")
        ws.default_compute_profile = body.default_compute_profile
    if body.name is not None:
        ws.name = body.name
    if body.budget_limit_usd is not None:
        ws.budget_limit_usd = body.budget_limit_usd
    if body.is_active is not None:
        ws.is_active = body.is_active

    db.commit()
    db.refresh(ws)

    record_audit_event(
        db,
        tenant_id=tenant.id,
        workspace_id=ws.id,
        actor_id=user.id,
        action="workspace.updated",
        resource_type="workspace",
        resource_id=ws.id,
        details=body.model_dump(exclude_none=True),
        request=request,
    )
    return _serialize(ws)


@router.delete("/{workspace_id}")
def delete_workspace(
    workspace_id: str,
    request: Request,
    user: User = Depends(require_admin),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    ws = (
        db.query(Workspace)
        .filter(Workspace.id == workspace_id, Workspace.tenant_id == tenant.id)
        .first()
    )
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace no encontrado.")
    if ws.slug == "default":
        raise HTTPException(status_code=400, detail="No se puede eliminar el workspace 'default'.")

    db.delete(ws)
    db.commit()

    record_audit_event(
        db,
        tenant_id=tenant.id,
        actor_id=user.id,
        action="workspace.deleted",
        resource_type="workspace",
        resource_id=workspace_id,
        details={"slug": ws.slug},
        request=request,
    )
    return {"message": "Workspace eliminado.", "id": workspace_id}
