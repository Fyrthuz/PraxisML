"""
API de Runtime Environments (entornos aislados con dependencias propias)
e introspección de capacidades pluggables de la plataforma.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.deps import get_current_tenant, require_editor, require_viewer
from app.platform.compute.base import get_compute_registry
from app.platform.compute.environment_runner import get_environment_runner_registry
from app.platform.events.event_bus import _event_bus_registry
from app.platform.data_engine.engine import get_data_engine_registry
from app.database import get_db
from app.models.runtime_environment import RuntimeEnvironment
from app.models.tenant import Tenant
from app.models.user import User
from app.platform.audit.service import record_audit_event
from app.platform.serving.runtimes import get_model_runtime_registry
from app.plugins.registry import get_plugin_registry
from app.platform.storage.service import get_storage_registry
from app.platform.tracker.backend import get_tracker_registry

router = APIRouter()


class EnvironmentCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=80)
    slug: Optional[str] = Field(default=None, max_length=64)
    description: Optional[str] = Field(default=None, max_length=300)
    workspace_id: Optional[str] = None
    python_version: str = Field(default="3.10")
    base_image: str = Field(default="python:3.10-slim")
    packages: List[str] = Field(default_factory=list)
    env_vars: Dict[str, str] = Field(default_factory=dict)
    compute_profile: str = Field(default="cpu-heavy")
    runner_type: str = Field(default="uv_isolated")


class EnvironmentUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=2, max_length=80)
    description: Optional[str] = Field(default=None, max_length=300)
    python_version: Optional[str] = None
    base_image: Optional[str] = None
    packages: Optional[List[str]] = None
    env_vars: Optional[Dict[str, str]] = None
    compute_profile: Optional[str] = None
    runner_type: Optional[str] = None


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "runtime-env"


def _serialize(env: RuntimeEnvironment) -> dict:
    return {
        "id": env.id,
        "tenant_id": env.tenant_id,
        "workspace_id": env.workspace_id,
        "name": env.name,
        "slug": env.slug,
        "description": env.description,
        "python_version": env.python_version,
        "base_image": env.base_image,
        "packages": env.packages or [],
        "env_vars": env.env_vars or {},
        "compute_profile": env.compute_profile,
        "runner_type": env.runner_type,
        "is_default": env.is_default,
        "created_at": env.created_at.isoformat() if env.created_at else None,
        "updated_at": env.updated_at.isoformat() if env.updated_at else None,
    }


def _ensure_default_environment(db: Session, tenant_id: str) -> list[RuntimeEnvironment]:
    items = (
        db.query(RuntimeEnvironment)
        .filter(RuntimeEnvironment.tenant_id == tenant_id)
        .order_by(RuntimeEnvironment.is_default.desc(), RuntimeEnvironment.created_at.asc())
        .all()
    )
    if not items:
        default_env = RuntimeEnvironment(
            tenant_id=tenant_id,
            name="Default ML Python 3.10",
            slug="default-ml-py310",
            description="Entorno base con pandas, numpy y scikit-learn.",
            python_version="3.10",
            base_image="python:3.10-slim",
            packages=["pandas", "numpy", "scikit-learn"],
            env_vars={},
            compute_profile="cpu-heavy",
            runner_type="uv_isolated",
            is_default=True,
        )
        db.add(default_env)
        db.commit()
        db.refresh(default_env)
        items = [default_env]
    return items


@router.get("/capabilities")
def get_platform_capabilities(
    _user: User = Depends(require_viewer),
    _tenant: Tenant = Depends(get_current_tenant),
):
    """Devuelve todos los backends, runtimes y plugins pluggables registrados en la instancia."""
    return {
        "storage_backends": get_storage_registry().list_backends(),
        "compute_backends": get_compute_registry().list_backends(),
        "environment_runners": get_environment_runner_registry().list_runners(),
        "serving_runtimes": get_model_runtime_registry().list_runtimes(),
        "event_bus_backends": _event_bus_registry.list_backends(),
        "data_engines": get_data_engine_registry().list_engines(),
        "tracker_backends": get_tracker_registry().list_backends(),
        "active_plugins": [p.to_metadata() for p in get_plugin_registry().list_plugins()],
    }


@router.get("/")
def list_environments(
    _user: User = Depends(require_viewer),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    items = _ensure_default_environment(db, tenant.id)
    return {"total": len(items), "items": [_serialize(e) for e in items]}


@router.post("/", status_code=201)
def create_environment(
    body: EnvironmentCreate,
    request: Request,
    user: User = Depends(require_editor),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    slug = _slugify(body.slug or body.name)
    exists = (
        db.query(RuntimeEnvironment)
        .filter(RuntimeEnvironment.tenant_id == tenant.id, RuntimeEnvironment.slug == slug)
        .first()
    )
    if exists:
        raise HTTPException(status_code=409, detail=f"Ya existe un entorno con slug '{slug}'.")

    env = RuntimeEnvironment(
        tenant_id=tenant.id,
        workspace_id=body.workspace_id,
        name=body.name,
        slug=slug,
        description=body.description,
        python_version=body.python_version,
        base_image=body.base_image,
        packages=[p.strip() for p in body.packages if p.strip()],
        env_vars=body.env_vars,
        compute_profile=body.compute_profile,
        runner_type=body.runner_type,
        created_by=user.id,
    )
    db.add(env)
    db.commit()
    db.refresh(env)

    record_audit_event(
        db,
        tenant_id=tenant.id,
        workspace_id=env.workspace_id,
        actor_id=user.id,
        action="environment.created",
        resource_type="runtime_environment",
        resource_id=env.id,
        details={"slug": env.slug, "packages": env.packages, "runner_type": env.runner_type},
        request=request,
    )
    return _serialize(env)


@router.patch("/{env_id}")
def update_environment(
    env_id: str,
    body: EnvironmentUpdate,
    request: Request,
    user: User = Depends(require_editor),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    env = (
        db.query(RuntimeEnvironment)
        .filter(RuntimeEnvironment.id == env_id, RuntimeEnvironment.tenant_id == tenant.id)
        .first()
    )
    if not env:
        raise HTTPException(status_code=404, detail="Entorno no encontrado.")

    for field_name, val in body.model_dump(exclude_none=True).items():
        if field_name == "packages":
            val = [p.strip() for p in val if p.strip()]
        setattr(env, field_name, val)

    env.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(env)

    record_audit_event(
        db,
        tenant_id=tenant.id,
        actor_id=user.id,
        action="environment.updated",
        resource_type="runtime_environment",
        resource_id=env.id,
        details=body.model_dump(exclude_none=True),
        request=request,
    )
    return _serialize(env)


@router.delete("/{env_id}")
def delete_environment(
    env_id: str,
    request: Request,
    user: User = Depends(require_editor),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    env = (
        db.query(RuntimeEnvironment)
        .filter(RuntimeEnvironment.id == env_id, RuntimeEnvironment.tenant_id == tenant.id)
        .first()
    )
    if not env:
        raise HTTPException(status_code=404, detail="Entorno no encontrado.")
    if env.is_default:
        raise HTTPException(status_code=400, detail="No se puede eliminar el entorno por defecto.")

    db.delete(env)
    db.commit()

    record_audit_event(
        db,
        tenant_id=tenant.id,
        actor_id=user.id,
        action="environment.deleted",
        resource_type="runtime_environment",
        resource_id=env_id,
        details={"slug": env.slug},
        request=request,
    )
    return {"message": "Entorno eliminado.", "id": env_id}
