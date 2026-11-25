"""
API de User-Defined Pipelines reutilizables y ejecutables sobre Runtime Environments.
Permite definir pipelines declarativos, código Python `def transform(df)` o híbridos,
versionarlos y ejecutarlos contra cualquier dataset del tenant.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
import time
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.deps import (
    get_current_tenant,
    get_storage_service,
    require_editor,
    require_viewer,
)
from app.compute.base import EnvironmentSpec
from app.compute.environment_runner import get_environment_runner_registry
from app.core_ml.preprocessing import apply_pipeline, build_pipeline, save_pipeline
from app.core_ml.tabular_parser import is_tabular, read_tabular
from app.database import get_db
from app.models.dataset import Dataset
from app.models.runtime_environment import RuntimeEnvironment
from app.models.tenant import Tenant
from app.models.user import User
from app.models.user_pipeline import UserPipeline
from app.platform.audit.service import record_audit_event
from app.services.storage_service import StorageService

logger = logging.getLogger(__name__)
router = APIRouter()

VALID_ENGINES = {"declarative", "python", "hybrid"}


class UserPipelineCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    slug: Optional[str] = Field(default=None, max_length=64)
    description: Optional[str] = Field(default=None, max_length=400)
    workspace_id: Optional[str] = None
    environment_id: Optional[str] = None
    engine: str = Field(default="declarative")  # declarative | python | hybrid
    steps: List[Dict[str, Any]] = Field(default_factory=list)
    custom_code: Optional[str] = None
    column_renames: Dict[str, str] = Field(default_factory=dict)
    target_column: Optional[str] = None


class UserPipelineUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=2, max_length=100)
    description: Optional[str] = Field(default=None, max_length=400)
    environment_id: Optional[str] = None
    engine: Optional[str] = None
    steps: Optional[List[Dict[str, Any]]] = None
    custom_code: Optional[str] = None
    column_renames: Optional[Dict[str, str]] = None
    target_column: Optional[str] = None


class PipelineRunRequest(BaseModel):
    dataset_id: str
    output_dataset_name: Optional[str] = None
    target_column: Optional[str] = None


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "pipeline"


def _serialize(pipe: UserPipeline, db: Session) -> dict:
    env = (
        db.query(RuntimeEnvironment).filter(RuntimeEnvironment.id == pipe.environment_id).first()
        if pipe.environment_id
        else None
    )
    return {
        "id": pipe.id,
        "tenant_id": pipe.tenant_id,
        "workspace_id": pipe.workspace_id,
        "environment_id": pipe.environment_id,
        "environment_name": env.name if env else None,
        "environment_packages": env.packages if env else [],
        "name": pipe.name,
        "slug": pipe.slug,
        "description": pipe.description,
        "engine": pipe.engine,
        "steps": pipe.steps or [],
        "custom_code": pipe.custom_code,
        "column_renames": pipe.column_renames or {},
        "target_column": pipe.target_column,
        "version": pipe.version,
        "created_at": pipe.created_at.isoformat() if pipe.created_at else None,
        "updated_at": pipe.updated_at.isoformat() if pipe.updated_at else None,
    }


@router.get("/")
def list_pipelines(
    _user: User = Depends(require_viewer),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    items = (
        db.query(UserPipeline)
        .filter(UserPipeline.tenant_id == tenant.id)
        .order_by(UserPipeline.updated_at.desc())
        .all()
    )
    return {"total": len(items), "items": [_serialize(p, db) for p in items]}


@router.post("/", status_code=201)
def create_pipeline(
    body: UserPipelineCreate,
    request: Request,
    user: User = Depends(require_editor),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    if body.engine not in VALID_ENGINES:
        raise HTTPException(status_code=400, detail=f"Engine inválido. Usa: {sorted(VALID_ENGINES)}")
    if body.engine in {"python", "hybrid"} and not (body.custom_code or "").strip():
        raise HTTPException(
            status_code=400,
            detail="Los pipelines de tipo 'python' o 'hybrid' requieren `custom_code` con `def transform(df): ...`.",
        )
    if body.engine == "declarative" and not body.steps:
        raise HTTPException(status_code=400, detail="Un pipeline declarativo requiere al menos un paso en `steps`.")

    if body.environment_id:
        env = (
            db.query(RuntimeEnvironment)
            .filter(
                RuntimeEnvironment.id == body.environment_id,
                RuntimeEnvironment.tenant_id == tenant.id,
            )
            .first()
        )
        if not env:
            raise HTTPException(status_code=404, detail="RuntimeEnvironment no encontrado.")

    base_slug = _slugify(body.slug or body.name)
    slug = base_slug
    idx = 2
    while (
        db.query(UserPipeline)
        .filter(UserPipeline.tenant_id == tenant.id, UserPipeline.slug == slug)
        .first()
    ):
        slug = f"{base_slug}-{idx}"
        idx += 1

    pipe = UserPipeline(
        tenant_id=tenant.id,
        workspace_id=body.workspace_id,
        environment_id=body.environment_id,
        name=body.name,
        slug=slug,
        description=body.description,
        engine=body.engine,
        steps=body.steps,
        custom_code=body.custom_code,
        column_renames=body.column_renames,
        target_column=body.target_column,
        created_by=user.id,
    )
    db.add(pipe)
    db.commit()
    db.refresh(pipe)

    record_audit_event(
        db,
        tenant_id=tenant.id,
        workspace_id=pipe.workspace_id,
        actor_id=user.id,
        action="pipeline.created",
        resource_type="user_pipeline",
        resource_id=pipe.id,
        details={"slug": pipe.slug, "engine": pipe.engine, "environment_id": pipe.environment_id},
        request=request,
    )
    return _serialize(pipe, db)


@router.patch("/{pipeline_id}")
def update_pipeline(
    pipeline_id: str,
    body: UserPipelineUpdate,
    request: Request,
    user: User = Depends(require_editor),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    pipe = (
        db.query(UserPipeline)
        .filter(UserPipeline.id == pipeline_id, UserPipeline.tenant_id == tenant.id)
        .first()
    )
    if not pipe:
        raise HTTPException(status_code=404, detail="Pipeline no encontrado.")

    updates = body.model_dump(exclude_none=True)
    if "engine" in updates and updates["engine"] not in VALID_ENGINES:
        raise HTTPException(status_code=400, detail="Engine inválido.")
    for k, v in updates.items():
        setattr(pipe, k, v)
    pipe.version = (pipe.version or 1) + 1
    pipe.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(pipe)

    record_audit_event(
        db,
        tenant_id=tenant.id,
        actor_id=user.id,
        action="pipeline.updated",
        resource_type="user_pipeline",
        resource_id=pipe.id,
        details={"version": pipe.version, "engine": pipe.engine},
        request=request,
    )
    return _serialize(pipe, db)


@router.delete("/{pipeline_id}")
def delete_pipeline(
    pipeline_id: str,
    request: Request,
    user: User = Depends(require_editor),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    pipe = (
        db.query(UserPipeline)
        .filter(UserPipeline.id == pipeline_id, UserPipeline.tenant_id == tenant.id)
        .first()
    )
    if not pipe:
        raise HTTPException(status_code=404, detail="Pipeline no encontrado.")

    db.delete(pipe)
    db.commit()
    record_audit_event(
        db,
        tenant_id=tenant.id,
        actor_id=user.id,
        action="pipeline.deleted",
        resource_type="user_pipeline",
        resource_id=pipeline_id,
        details={"slug": pipe.slug},
        request=request,
    )
    return {"message": "Pipeline eliminado.", "id": pipeline_id}


@router.post("/{pipeline_id}/run")
def run_pipeline(
    pipeline_id: str,
    body: PipelineRunRequest,
    request: Request,
    user: User = Depends(require_editor),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
    storage: StorageService = Depends(get_storage_service),
):
    """Ejecuta un UserPipeline sobre un dataset usando su RuntimeEnvironment configurado."""
    pipe = (
        db.query(UserPipeline)
        .filter(UserPipeline.id == pipeline_id, UserPipeline.tenant_id == tenant.id)
        .first()
    )
    if not pipe:
        raise HTTPException(status_code=404, detail="Pipeline no encontrado.")

    dataset = (
        db.query(Dataset)
        .filter(Dataset.id == body.dataset_id, Dataset.tenant_id == tenant.id)
        .first()
    )
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset no encontrado.")
    if not dataset.file_type or not is_tabular(dataset.file_type):
        raise HTTPException(status_code=400, detail="El dataset debe ser tabular.")

    data_bytes = storage.download(dataset.file_path)
    df = read_tabular(BytesIO(data_bytes), dataset.file_type)
    target_col = body.target_column or pipe.target_column

    pipeline_artifact_uri: Optional[str] = None
    exec_stdout = ""
    df_out = df
    t0 = time.perf_counter()

    try:
        # 1. Ejecutar parte declarativa (si existe)
        if pipe.engine in {"declarative", "hybrid"} and pipe.steps:
            cfg = {
                "steps": pipe.steps,
                "target_column": target_col,
                "column_renames": pipe.column_renames or {},
            }
            ct = build_pipeline(cfg, df_out.columns.tolist())
            df_out, y = apply_pipeline(ct, df_out, target_col, column_renames=pipe.column_renames)
            if y is not None and target_col:
                df_out[target_col] = y.values
            pipeline_artifact_uri = save_pipeline(ct, f"{pipe.slug}_v{pipe.version}", tenant.id, cfg)

        # 2. Ejecutar código Python personalizado en su RuntimeEnvironment aislado
        if pipe.engine in {"python", "hybrid"} and pipe.custom_code:
            env_record = (
                db.query(RuntimeEnvironment)
                .filter(
                    RuntimeEnvironment.id == pipe.environment_id,
                    RuntimeEnvironment.tenant_id == tenant.id,
                )
                .first()
                if pipe.environment_id
                else None
            )
            env_spec = EnvironmentSpec(
                name=env_record.name if env_record else "default",
                python_version=env_record.python_version if env_record else "3.10",
                base_image=env_record.base_image if env_record else "python:3.10-slim",
                packages=list(env_record.packages or []) if env_record else [],
                env_vars=dict(env_record.env_vars or {}) if env_record else {},
                runner_type=env_record.runner_type if env_record else "uv_isolated",
            )
            runner = get_environment_runner_registry().create(env_spec.runner_type)
            exec_result = runner.run_dataframe_transform(df_out, pipe.custom_code, env_spec)
            df_out = exec_result.dataframe
            exec_stdout = exec_result.stdout

            if pipe.column_renames:
                df_out = df_out.rename(columns=pipe.column_renames)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error ejecutando pipeline: {exc}")

    # 3. Guardar nuevo dataset transformado
    existing_count = (
        db.query(Dataset)
        .filter(Dataset.tenant_id == tenant.id, Dataset.name == dataset.name)
        .count()
    )
    new_version = existing_count + 1
    out_name = (body.output_dataset_name or "").strip() or f"{dataset.name} [{pipe.slug} v{pipe.version}]"
    storage_key = f"tenants/{tenant.id}/datasets/{pipe.slug}_v{pipe.version}_{new_version}.csv"

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            df_out.to_csv(tmp.name, index=False)
            tmp_path = tmp.name
        with open(tmp_path, "rb") as f:
            content = f.read()
        storage.upload(storage_key, content)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    new_ds = Dataset(
        name=out_name,
        description=f"Generado por pipeline '{pipe.name}' (v{pipe.version}, engine={pipe.engine}) desde '{dataset.name}'.",
        file_path=storage_key,
        file_size_bytes=len(content),
        file_type="csv",
        num_rows=len(df_out),
        num_columns=len(df_out.columns),
        column_names=df_out.columns.tolist(),
        version=new_version,
        pipeline_path=pipeline_artifact_uri,
        tenant_id=tenant.id,
        parent_dataset_id=dataset.id,
    )
    db.add(new_ds)
    try:
        from app.observability.usage_collector import get_usage_collector

        get_usage_collector().record_pipeline_run(
            db=db,
            tenant_id=tenant.id,
            pipeline_id=pipe.id,
            engine=pipe.engine,
            duration_s=round(time.perf_counter() - t0, 4),
            success=True,
            tags={"slug": pipe.slug, "output_rows": len(df_out)},
        )
    except Exception:
        pass
    db.commit()
    db.refresh(new_ds)

    record_audit_event(
        db,
        tenant_id=tenant.id,
        workspace_id=pipe.workspace_id,
        actor_id=user.id,
        action="pipeline.executed",
        resource_type="user_pipeline",
        resource_id=pipe.id,
        details={
            "source_dataset_id": dataset.id,
            "output_dataset_id": new_ds.id,
            "shape": [len(df_out), len(df_out.columns)],
        },
        request=request,
    )

    return {
        "message": "Pipeline ejecutado correctamente.",
        "pipeline_id": pipe.id,
        "output_dataset_id": new_ds.id,
        "output_dataset_name": new_ds.name,
        "transformed_shape": [len(df_out), len(df_out.columns)],
        "transformed_columns": df_out.columns.tolist(),
        "pipeline_path": pipeline_artifact_uri,
        "stdout": exec_stdout,
    }
