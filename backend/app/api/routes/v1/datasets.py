from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List, Optional
import os
import shutil
import logging

from app.services.dvc_service import DVCService
from app.database import get_db
from app.models.dataset import Dataset
from app.models.user import User
from app.models.tenant import Tenant
from app.schemas.dataset import DatasetResponse, DatasetPreviewResponse
from app.api.deps import (
    get_current_tenant,
    require_editor,
    require_viewer,
    require_admin,
    check_dataset_quota,
)
from app.core.config import settings
from app.core_ml.tabular_parser import (
    detect_file_type,
    is_tabular,
    extract_metadata,
    get_preview,
)

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/{dataset_id}/download")
def download_dataset(
    dataset_id: str,
    _user: User = Depends(require_viewer),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """
    Descarga un dataset específico. Si está trackeado con DVC, asegura que esté en disco local.
    """
    dataset = (
        db.query(Dataset)
        .filter(Dataset.id == dataset_id, Dataset.tenant_id == tenant.id)
        .first()
    )

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset no encontrado")

    file_path = dataset.file_path

    # Si es DVC, intentamos hacer pull por si acaso no está en el nodo actual
    if dataset.is_dvc_tracked:
        try:
            dvc_service = DVCService(tenant.id)
            # Pull ensure the file exists locally
            dvc_service.pull_dataset(file_path)
        except Exception as e:
            logger.warning(f"Failed to pull dataset from DVC before download: {e}")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Archivo no encontrado en el servidor")

    return FileResponse(
        path=file_path,
        filename=os.path.basename(file_path),
        media_type="application/octet-stream"
    )
# Extensiones aceptadas
_ALLOWED_EXTENSIONS = (".csv", ".xlsx", ".parquet", ".zip")


# ──────────────────────────────────────────────────────────────────────────────
# POST  /datasets/  — Upload (multi-formato)
# ──────────────────────────────────────────────────────────────────────────────
@router.post("/", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    name: str = Form(...),
    description: str = Form(None),
    file: UploadFile = File(...),
    config_file: Optional[UploadFile] = File(None),
    is_dvc_tracked: bool = Form(default=False),
    dvc_registry_name: Optional[str] = Form(default=None),
    _user: User = Depends(require_editor),
    tenant: Tenant = Depends(check_dataset_quota),
    db: Session = Depends(get_db),
):
    """
    Sube un dataset en formato .csv, .xlsx, .parquet o .zip (imágenes).
    Para archivos tabulares, se extrae metadata automáticamente (filas, columnas, schema).

    Requiere rol **editor** o superior.
    Se valida la cuota de datasets del tenant.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="El archivo debe tener un nombre.")

    # Validar extensión
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Formato no soportado: {ext}. Formatos aceptados: {', '.join(_ALLOWED_EXTENSIONS)}",
        )

    file_type = detect_file_type(file.filename)

    # ── Guardar en disco ─────────────────────────────────────────────────────
    tenant_dataset_dir = os.path.join(
        settings.DATA_DIR, "tenants", tenant.id, "datasets"
    )
    os.makedirs(tenant_dataset_dir, exist_ok=True)

    file_path = os.path.join(tenant_dataset_dir, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error guardando el archivo: {e}",
        )

    # ── Config file (opcional, para .zip con imágenes) ───────────────────────
    config_path = None
    if config_file and config_file.filename:
        if not config_file.filename.endswith(".json"):
            raise HTTPException(status_code=400, detail="Config file must be a JSON")
        config_path = os.path.join(
            tenant_dataset_dir, f"{name}_config_{config_file.filename}"
        )
        try:
            with open(config_path, "wb") as buffer:
                shutil.copyfileobj(config_file.file, buffer)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error saving config file: {e}"
            )

    file_size = os.path.getsize(file_path)

    # ── Extraer metadata tabular ─────────────────────────────────────────────
    num_rows = None
    num_columns = None
    column_names = None

    if file_type and is_tabular(file_type):
        try:
            meta = extract_metadata(file_path, file_type)
            num_rows = meta["num_rows"]
            num_columns = meta["num_columns"]
            column_names = meta["column_names"]
            logger.info(
                "Metadata extraída para '%s': %d filas × %d columnas",
                file.filename,
                num_rows,
                num_columns,
            )
        except Exception as e:
            logger.warning("No se pudo extraer metadata tabular: %s", e)

    # ── Calcular versión ─────────────────────────────────────────────────────
    existing_count = (
        db.query(Dataset)
        .filter(Dataset.tenant_id == tenant.id, Dataset.name == name)
        .count()
    )
    version = existing_count + 1

    # ── DVC Tracking (Opcional) ─────────────────────────────────────────────
    _is_actually_tracked = False
    dvc_hash = None
    dvc_remote = None
    dvc_registry = dvc_registry_name or f"tenant_{tenant.id}_{name.replace(' ', '_')}"

    if is_dvc_tracked:
        try:

            dvc_service = DVCService(tenant.id)
            dvc_service.init_repository()
            dvc_service.configure_remote()

            dvc_result = dvc_service.add_dataset(file_path, dvc_registry)
            _is_actually_tracked = True
            dvc_hash = dvc_result.get("hash")
            dvc_remote = "minio"
            logger.info(f"Dataset {name} tracked with DVC, hash: {dvc_hash}")
        except Exception as e:
            logger.warning(f"Failed to track dataset with DVC: {e}")

    # ── Registrar en BD ──────────────────────────────────────────────────────
    new_dataset = Dataset(
        name=name,
        description=description,
        file_path=file_path,
        config_path=config_path,
        file_size_bytes=file_size,
        file_type=file_type,
        num_rows=num_rows,
        num_columns=num_columns,
        column_names=column_names,
        version=version,
        tenant_id=tenant.id,
        is_dvc_tracked=_is_actually_tracked,
        dvc_hash=dvc_hash,
        dvc_remote=dvc_remote,
        dvc_registry_name=dvc_registry,
        dvc_version=version if _is_actually_tracked else None,
    )

    db.add(new_dataset)
    db.commit()
    db.refresh(new_dataset)
    return new_dataset


# ──────────────────────────────────────────────────────────────────────────────
# GET  /datasets/  — Listar datasets del tenant
# ──────────────────────────────────────────────────────────────────────────────
@router.get("/", response_model=List[DatasetResponse])
def get_datasets(
    _user: User = Depends(require_viewer),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """Obtiene todos los datasets del tenant (multi-tenant RLS). Requiere rol **viewer** o superior."""
    datasets = db.query(Dataset).filter(Dataset.tenant_id == tenant.id).all()
    return datasets


# ──────────────────────────────────────────────────────────────────────────────
# GET  /datasets/{dataset_id}/preview  — Preview de datos tabulares
# ──────────────────────────────────────────────────────────────────────────────
@router.get("/{dataset_id}/preview", response_model=DatasetPreviewResponse)
def preview_dataset(
    dataset_id: str,
    max_rows: int = 20,
    _user: User = Depends(require_viewer),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """
    Devuelve las primeras N filas de un dataset tabular como JSON.
    Solo funciona para archivos .csv, .xlsx, .parquet.
    Requiere rol **viewer** o superior.
    """
    dataset = (
        db.query(Dataset)
        .filter(Dataset.id == dataset_id, Dataset.tenant_id == tenant.id)
        .first()
    )
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset no encontrado.")

    if not dataset.file_type or not is_tabular(dataset.file_type):
        raise HTTPException(
            status_code=400,
            detail=f"Preview solo disponible para datasets tabulares. Tipo actual: {dataset.file_type}",
        )

    try:
        preview_df, meta = get_preview(
            dataset.file_path, dataset.file_type, max_rows=max_rows
        )
        # Reemplazar NaN con None para serialización JSON
        preview_rows = preview_df.where(preview_df.notna(), None).to_dict(
            orient="records"
        )
        return DatasetPreviewResponse(
            dataset_id=dataset.id,
            file_type=dataset.file_type,
            num_rows=meta["num_rows"],
            num_columns=meta["num_columns"],
            column_names=meta["column_names"],
            column_dtypes=meta["column_dtypes"],
            preview_rows=preview_rows,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al generar preview: {e}",
        )


# ──────────────────────────────────────────────────────────────────────────────
# DELETE  /datasets/{dataset_id}  — Eliminar dataset
# ──────────────────────────────────────────────────────────────────────────────
@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset(
    dataset_id: str,
    _user: User = Depends(require_admin),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """
    Elimina un dataset: borra el archivo de disco, el config file si existe,
    y el registro de la BD. No elimina predicciones asociadas (linaje).
    Requiere rol **admin**.
    """
    dataset = (
        db.query(Dataset)
        .filter(Dataset.id == dataset_id, Dataset.tenant_id == tenant.id)
        .first()
    )
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset no encontrado.")

    # Borrar archivos de disco
    if dataset.file_path and os.path.exists(dataset.file_path):
        try:
            os.remove(dataset.file_path)
            logger.info("Archivo eliminado: %s", dataset.file_path)
        except OSError as e:
            logger.warning("No se pudo eliminar archivo: %s — %s", dataset.file_path, e)

    if dataset.config_path and os.path.exists(dataset.config_path):
        try:
            os.remove(dataset.config_path)
        except OSError as e:
            logger.warning(
                "No se pudo eliminar config: %s — %s", dataset.config_path, e
            )

    # Borrar registro de BD
    db.delete(dataset)
    db.commit()

    return None


# ──────────────────────────────────────────────────────────────────────────────
# DVC Dataset Registry Endpoints
# ──────────────────────────────────────────────────────────────────────────────


@router.get("/registry", response_model=List[dict])
def list_dataset_registries(
    _user: User = Depends(require_viewer),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """
    Lista todos los datasets que están siendo versionados con DVC.
    """
    datasets = (
        db.query(Dataset)
        .filter(Dataset.tenant_id == tenant.id, Dataset.is_dvc_tracked)
        .all()
    )

    registries = {}
    tenant_prefix = f"tenant_{tenant.id}_"

    for ds in datasets:
        reg_name = ds.dvc_registry_name
        if reg_name:
            if reg_name not in registries:
                display_name = reg_name
                if reg_name.startswith(tenant_prefix):
                    display_name = reg_name[len(tenant_prefix):]

                registries[reg_name] = {
                    "name": reg_name,
                    "display_name": display_name,
                    "datasets": [],
                    "versions": 0,
                }
            registries[reg_name]["datasets"].append(
                {
                    "id": ds.id,
                    "name": ds.name,
                    "version": ds.version,
                    "dvc_hash": ds.dvc_hash,
                    "dvc_version": ds.dvc_version,
                }
            )
            registries[reg_name]["versions"] = max(
                registries[reg_name]["versions"], ds.dvc_version or 0
            )

    return list(registries.values())


@router.get("/registry/{registry_name}/versions")
def get_dataset_versions(
    registry_name: str,
    _user: User = Depends(require_viewer),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """
    Obtiene todas las versiones de un dataset registry específico.
    """
    datasets = (
        db.query(Dataset)
        .filter(
            Dataset.tenant_id == tenant.id, Dataset.dvc_registry_name == registry_name
        )
        .order_by(Dataset.version.desc())
        .all()
    )

    return {
        "registry_name": registry_name,
        "versions": [
            {
                "id": ds.id,
                "name": ds.name,
                "version": ds.version,
                "dvc_version": ds.dvc_version,
                "dvc_hash": ds.dvc_hash,
                "created_at": ds.created_at,
                "file_size_bytes": ds.file_size_bytes,
            }
            for ds in datasets
        ],
    }


@router.post("/{dataset_id}/promote")
def promote_dataset(
    dataset_id: str,
    _user: User = Depends(require_editor),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """
    Promociona un dataset a 'Production' (marcarlo como el dataset activo).
    """
    dataset = (
        db.query(Dataset)
        .filter(Dataset.id == dataset_id, Dataset.tenant_id == tenant.id)
        .first()
    )

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset no encontrado")

    # Get all datasets with same registry and set them to non-production
    if dataset.dvc_registry_name:
        other_datasets = (
            db.query(Dataset)
            .filter(
                Dataset.tenant_id == tenant.id,
                Dataset.dvc_registry_name == dataset.dvc_registry_name,
                Dataset.id != dataset_id,
                Dataset.is_active,
            )
            .all()
        )

        for ds in other_datasets:
            ds.is_active = False

    dataset.is_active = True
    db.commit()

    return {
        "message": "Dataset promoted to production",
        "dataset_id": dataset.id,
        "is_active": dataset.is_active,
    }


@router.post("/{dataset_id}/dvc/push")
def push_dataset_to_remote(
    dataset_id: str,
    _user: User = Depends(require_editor),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """
    Sube un dataset al remote de DVC.
    """
    dataset = (
        db.query(Dataset)
        .filter(Dataset.id == dataset_id, Dataset.tenant_id == tenant.id)
        .first()
    )

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset no encontrado")

    if not dataset.is_dvc_tracked:
        raise HTTPException(status_code=400, detail="Dataset no está trackeado con DVC")

    try:
        from app.services.dvc_service import DVCService

        dvc_service = DVCService(tenant.id)
        success = dvc_service.push_dataset(dataset.file_path)

        if success:
            return {"message": "Dataset subido a DVC remote"}
        else:
            raise HTTPException(status_code=500, detail="Error al subir a DVC remote")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")


@router.post("/{dataset_id}/dvc/pull")
def pull_dataset_from_remote(
    dataset_id: str,
    _user: User = Depends(require_editor),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """
    Descarga un dataset del remote de DVC.
    """
    dataset = (
        db.query(Dataset)
        .filter(Dataset.id == dataset_id, Dataset.tenant_id == tenant.id)
        .first()
    )

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset no encontrado")

    if not dataset.is_dvc_tracked:
        raise HTTPException(status_code=400, detail="Dataset no está trackeado con DVC")

    try:
        from app.services.dvc_service import DVCService

        dvc_service = DVCService(tenant.id)
        success = dvc_service.pull_dataset(dataset.file_path)

        if success:
            return {"message": "Dataset descargado de DVC remote"}
        else:
            raise HTTPException(
                status_code=500, detail="Error al descargar de DVC remote"
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")
