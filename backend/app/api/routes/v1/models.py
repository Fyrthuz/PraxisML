import csv
import logging
import os
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import mlflow
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.api.deps import (
    check_model_quota,
    get_current_tenant,
    require_admin,
    require_editor,
    require_viewer,
)
from app.core.config import settings
from app.database import get_db
from app.models.ml_model import MLModel
from app.models.tenant import Tenant
from app.models.user import User
from app.schemas.ml_model import MLModelCreate, MLModelResponse
from app.schemas.pagination import PaginatedResponse
from app.services.mlflow_service import MLFlowService

router = APIRouter()
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Register model (manual — provide an existing MLFlow run_id)
# ──────────────────────────────────────────────────────────────────────────────


@router.post("/", response_model=MLModelResponse, status_code=status.HTTP_201_CREATED)
def register_model(
    model_in: MLModelCreate,
    _user: User = Depends(require_editor),
    tenant: Tenant = Depends(check_model_quota),
    db: Session = Depends(get_db),
):
    """
    Registra en la BD un modelo cuyo run_id de MLFlow ya existe.
    Útil si el modelo se entrenó externamente y se subió a MLFlow a mano.

    Requiere rol **editor** o superior. Valida cuota de modelos del tenant.
    """

    existing = (
        db.query(MLModel)
        .filter(MLModel.mlflow_run_id == model_in.mlflow_run_id)
        .first()
    )
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Este run_id de MLFlow ya está registrado en la base de datos.",
        )

    # Try to register in MLflow Model Registry (only if we have a run_id)
    mlflow_registry_name = None
    mlflow_version = None
    if model_in.mlflow_run_id:
        try:
            mlflow_svc = MLFlowService()
            model_name_for_registry = (
                f"tenant_{tenant.id}_{model_in.name.replace(' ', '_')}"
            )
            registry_result = mlflow_svc.register_model_to_registry(
                model_name=model_name_for_registry,
                run_id=model_in.mlflow_run_id,
                description=model_in.description or "",
            )
            mlflow_registry_name = registry_result["name"]
            mlflow_version = str(registry_result["version"])
        except Exception as e:
            logger.warning("Failed to register in MLflow Registry: %s", str(e))

    new_model = MLModel(
        **model_in.model_dump(),
        tenant_id=tenant.id,
        mlflow_registry_name=mlflow_registry_name,
        mlflow_version=mlflow_version,
    )
    db.add(new_model)
    db.commit()
    db.refresh(new_model)
    return new_model


# ──────────────────────────────────────────────────────────────────────────────
# Upload .pth → register in MLFlow → save to DB  (el flujo integrado)
# ──────────────────────────────────────────────────────────────────────────────


@router.post(
    "/upload", response_model=MLModelResponse, status_code=status.HTTP_201_CREATED
)
def upload_and_register_model(
    name: str = Form(...),
    description: Optional[str] = Form(None),
    architecture: str = Form("unknown"),
    num_classes: int = Form(2),
    is_public: bool = Form(False),
    registry_name: Optional[str] = Form(default=None),
    file: UploadFile = File(...),
    _user: User = Depends(require_editor),
    tenant: Tenant = Depends(check_model_quota),
    db: Session = Depends(get_db),
):
    """
    Sube un fichero .pth, lo registra en MLFlow automáticamente y crea el
    registro en la base de datos.  Todo en un único paso.

    Requiere rol **editor** o superior. Valida cuota de modelos del tenant.

    Si se proporciona registry_name, el modelo se registrará en ese modelo del
    MLflow Model Registry. Si no, se creará uno nuevo con el nombre del modelo.
    """

    if not file.filename or not file.filename.endswith((".pth", ".pt", ".ts")):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Solo se aceptan ficheros .pth (estado clásico) o .pt/.ts (TorchScript).",
        )

    is_torchscript = file.filename.endswith((".pt", ".ts"))

    # ── Guardar en disco ────────────────────────────────────────────────────
    models_dir = Path(settings.DATA_DIR) / "tenants" / tenant.id / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    pth_path = models_dir / file.filename
    with pth_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # ── Registrar en MLFlow (Si NO es TorchScript) ──────────────────────────
    run_id = None
    mlflow_registry_name = None
    mlflow_version = None
    if not is_torchscript:
        try:
            mlflow_svc = MLFlowService()
            run_id = mlflow_svc.register_pth_model(
                pth_path=str(pth_path),
                model_name=name,
                tenant_id=tenant.id,
                architecture=architecture,
                num_classes=num_classes,
            )

            # Register to MLflow Model Registry
            try:
                # Use provided registry name (ensuring prefix) or create one based on model name
                if registry_name:
                    model_name_for_registry = registry_name
                    if not model_name_for_registry.startswith(f"tenant_{tenant.id}_"):
                        model_name_for_registry = (
                            f"tenant_{tenant.id}_{model_name_for_registry}"
                        )
                else:
                    model_name_for_registry = (
                        f"tenant_{tenant.id}_{name.replace(' ', '_')}"
                    )

                # Create registry if it doesn't exist
                try:
                    mlflow_svc.create_registered_model(
                        name=model_name_for_registry,
                        description=description or f"Model: {name}",
                    )
                except Exception:
                    pass  # Already exists

                # Register model version to registry
                registry_result = mlflow_svc.register_model_to_registry(
                    model_name=model_name_for_registry,
                    run_id=run_id,
                    description=description or "",
                )
                mlflow_registry_name = registry_result["name"]
                mlflow_version = str(registry_result["version"])
            except Exception as e:
                logger.warning("Failed to register in MLflow Registry: %s", str(e))

        except Exception as exc:
            pth_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error al registrar en MLFlow: {exc}",
            )

    # ── Crear registro en BD ─────────────────────────────────────────────────
    new_model = MLModel(
        name=name,
        description=description,
        mlflow_run_id=run_id,
        mlflow_registry_name=mlflow_registry_name,
        mlflow_version=mlflow_version,
        metrics_metadata={
            "architecture": architecture,
            "num_classes": num_classes,
            "source_file": file.filename,
        },
        is_public=is_public,
        is_torchscript=is_torchscript,
        torchscript_path=str(pth_path) if is_torchscript else None,
        tenant_id=tenant.id,
    )
    db.add(new_model)
    db.commit()
    db.refresh(new_model)
    return new_model


# ──────────────────────────────────────────────────────────────────────────────
# List models for a tenant (own + public)
# ──────────────────────────────────────────────────────────────────────────────


@router.get("/", response_model=PaginatedResponse[MLModelResponse])
def get_models(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    _user: User = Depends(require_viewer),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """
    Obtener modelos del tenant MÁS los modelos base disponibles globalmente con paginación.
    Requiere rol **viewer** o superior.
    """
    query = db.query(MLModel).filter(
        (MLModel.tenant_id == tenant.id) | (MLModel.is_public is True)
    )

    total = query.count()
    total_pages = (total + page_size - 1) // page_size if total > 0 else 0

    models = query.offset((page - 1) * page_size).limit(page_size).all()

    return PaginatedResponse(
        items=models,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
        has_next=page < total_pages,
        has_prev=page > 1,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Delete model
# ──────────────────────────────────────────────────────────────────────────────


@router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_model(
    model_id: str,
    _user: User = Depends(require_admin),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """
    Elimina un modelo de la base de datos que pertenezca al tenant actual.
    Requiere rol **admin**.
    (Opcionalmente también se podrían purgar sus artefactos en disco/MLFlow).
    """
    model = (
        db.query(MLModel)
        .filter(MLModel.id == model_id, MLModel.tenant_id == tenant.id)
        .first()
    )

    if not model:
        raise HTTPException(
            status_code=404, detail="Model no encontrado o sin permisos."
        )

    # Opcional: Eliminar archivos .pth/.pt de disco si existen
    if model.torchscript_path and os.path.exists(model.torchscript_path):
        try:
            os.remove(model.torchscript_path)
        except Exception as e:
            logger.warning(
                "Could not delete torchscript file %s: %s",
                model.torchscript_path,
                str(e),
            )

    # Intentar eliminar de MLFlow
    if model.mlflow_run_id:
        try:
            mlflow_svc = MLFlowService()
            client = mlflow.tracking.MlflowClient(
                tracking_uri=mlflow_svc.get_tracking_uri()
            )
            client.delete_run(model.mlflow_run_id)
        except Exception as e:
            logger.warning(
                "Failed to delete MLflow run %s: %s", model.mlflow_run_id, str(e)
            )

    # Eliminar predicciones asociadas para evitar ForeignKeyViolation
    from app.models.prediction import Prediction

    predictions = db.query(Prediction).filter(Prediction.model_id == model_id).all()
    for pred in predictions:
        # Borrar archivos .npy asociados si existen
        if pred.result_path:
            full_result_path = os.path.join(settings.DATA_DIR, pred.result_path)
            if os.path.exists(full_result_path):
                try:
                    os.remove(full_result_path)
                except Exception:
                    pass
        if pred.uncertainty_path:
            full_uncert_path = os.path.join(settings.DATA_DIR, pred.uncertainty_path)
            if os.path.exists(full_uncert_path):
                try:
                    os.remove(full_uncert_path)
                except Exception:
                    pass
        db.delete(pred)

    db.delete(model)
    db.commit()
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Download Model (.zip with CSV info and MLFlow artifacts)
# ──────────────────────────────────────────────────────────────────────────────
# Model Download Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _create_model_zip_response(
    run_id: str,
    name: str,
    description: str = "",
    metrics_metadata: Optional[dict] = None,
    preprocessing_pipeline_path: Optional[str] = None,
    torchscript_path: Optional[str] = None,
):
    """
    Helper para crear un zip con los artefactos de un run de MLflow y metadatos.
    """
    # Crear directorio temporal para preparar el zip
    temp_dir = tempfile.mkdtemp(prefix="mlmodel_")

    try:
        # 1. Crear el CSV con metadatos
        csv_path = os.path.join(temp_dir, "model_info.csv")
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Key", "Value"])
            writer.writerow(["Model Name", name])
            writer.writerow(["Description", description or ""])
            writer.writerow(["MLFlow Run ID", run_id or ""])
            writer.writerow(["Created At", str(datetime.now())])

            # Agregar metadatos e hiperparámetros
            if metrics_metadata:
                for k, v in metrics_metadata.items():
                    if isinstance(v, (list, dict)):
                        import json

                        v = json.dumps(v)
                    writer.writerow([f"metadata_{k}", str(v)])

        # 2. Descargar artefactos de MLflow
        if run_id:
            try:
                mlflow_svc = MLFlowService()
                client = mlflow.tracking.MlflowClient(
                    tracking_uri=mlflow_svc.get_tracking_uri()
                )
                artifact_path = client.download_artifacts(run_id=run_id, path="")

                # Copiar contenidos del directorio de artefactos
                dest_artifacts = os.path.join(temp_dir, "artifacts")
                shutil.copytree(artifact_path, dest_artifacts, dirs_exist_ok=True)
            except Exception as e:
                logger.warning(
                    "Could not download artifacts for run %s: %s", run_id, str(e)
                )

        # 3. TorchScript/Pth manual
        if torchscript_path and os.path.exists(torchscript_path):
            shutil.copy2(torchscript_path, temp_dir)

        # 4. Descargar pipeline de preprocesamiento (si existe)
        if preprocessing_pipeline_path and preprocessing_pipeline_path.startswith(
            "runs:/"
        ):
            try:
                pipeline_parts = preprocessing_pipeline_path.split("/")
                pipeline_run_id = pipeline_parts[1]
                pipeline_file_path = "/".join(pipeline_parts[2:])

                mlflow_svc = MLFlowService()
                client = mlflow.tracking.MlflowClient(
                    tracking_uri=mlflow_svc.get_tracking_uri()
                )
                downloaded_pipe_path = client.download_artifacts(
                    run_id=pipeline_run_id, path=pipeline_file_path
                )

                if os.path.isfile(downloaded_pipe_path):
                    shutil.copy2(
                        downloaded_pipe_path,
                        os.path.join(temp_dir, "preprocessing_pipeline.joblib"),
                    )
                elif os.path.isdir(downloaded_pipe_path):
                    shutil.copytree(
                        downloaded_pipe_path,
                        os.path.join(temp_dir, "preprocessing_pipeline"),
                        dirs_exist_ok=True,
                    )
            except Exception as e:
                logger.warning("Could not download preprocessing pipeline: %s", str(e))

        # 5. Crear archivo Zip
        # Nota: Usamos zip_path en temp_dir para evitar conflictos
        zip_path = tempfile.mktemp(suffix=".zip", prefix=f"{name.replace(' ', '_')}_")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, _dirs, files in os.walk(temp_dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    arcname = os.path.relpath(full_path, temp_dir)
                    zipf.write(full_path, arcname)

        return FileResponse(
            path=zip_path,
            filename=f"{name.replace(' ', '_')}.zip",
            media_type="application/zip",
        )

    finally:
        # No podemos borrar el temp_dir aquí porque FileResponse lo necesita
        # FastAPI se encarga de servir el zip_path.
        # Idealmente usaríamos background tasks para limpieza posterior.
        pass


@router.get("/{model_id}/download", response_class=FileResponse)
def download_model(
    model_id: str,
    tenant_id: str,
    _user: User = Depends(require_viewer),
    db: Session = Depends(get_db),
):
    """
    Descarga el modelo en formato .zip.
    Incluye los artefactos almacenados en MLFlow y un CSV con la información
    y los hiperparámetros del modelo.

    Requiere rol **viewer** o superior.
    """
    model = (
        db.query(MLModel)
        .filter(
            MLModel.id == model_id,
            (MLModel.tenant_id == tenant_id) | (MLModel.is_public is True),
        )
        .first()
    )

    if not model:
        raise HTTPException(status_code=404, detail="Model no encontrado.")

    return _create_model_zip_response(
        run_id=model.mlflow_run_id,
        name=model.name,
        description=model.description,
        metrics_metadata=model.metrics_metadata,
        preprocessing_pipeline_path=model.preprocessing_pipeline_path,
        torchscript_path=model.torchscript_path
        if hasattr(model, "is_torchscript") and model.is_torchscript
        else None,
    )


@router.get("/runs/{run_id}/download", response_class=FileResponse)
def download_model_by_run(
    run_id: str,
    _user: User = Depends(require_viewer),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """
    Descarga un modelo basándose en su run_id de MLflow.
    Intenta enriquecerlo con datos si existe en nuestra BD.
    """
    model = (
        db.query(MLModel)
        .filter(MLModel.mlflow_run_id == run_id, MLModel.tenant_id == tenant.id)
        .first()
    )

    if model:
        return _create_model_zip_response(
            run_id=model.mlflow_run_id,
            name=model.name,
            description=model.description,
            metrics_metadata=model.metrics_metadata,
            preprocessing_pipeline_path=model.preprocessing_pipeline_path,
        )
    else:
        # Si no existe en BD, descargamos lo básico de MLflow
        try:
            mlflow_svc = MLFlowService()
            run_details = mlflow_svc.get_run_details(run_id)
            name = run_details.get("tags", {}).get(
                "mlflow.runName", f"run_{run_id[:8]}"
            )

            return _create_model_zip_response(
                run_id=run_id,
                name=name,
                description="Downloaded from MLflow Registry",
                metrics_metadata={
                    "metrics": run_details.get("metrics"),
                    "params": run_details.get("params"),
                    "tags": run_details.get("tags"),
                },
            )
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error downloading run artifacts: {e}"
            )


# ──────────────────────────────────────────────────────────────────────────────
# MLFlow info (tracking URI + experiment list)
# ──────────────────────────────────────────────────────────────────────────────


@router.get("/mlflow-info")
def mlflow_info(_user: User = Depends(require_viewer)):
    """
    Devuelve la URI de tracking activa y la lista de experimentos en MLFlow.
    Útil para el frontend/dashboard.
    Requiere rol **viewer** o superior.
    """
    try:
        mlflow_svc = MLFlowService()
        tracking_uri = mlflow_svc.get_tracking_uri()
        client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
        experiments = [
            {
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "lifecycle_stage": exp.lifecycle_stage,
            }
            for exp in client.search_experiments()
        ]
        return {
            "tracking_uri": tracking_uri,
            "experiments": experiments,
            "ui_url": "http://localhost:5001",
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al conectar con MLFlow: {exc}",
        )


# ──────────────────────────────────────────────────────────────────────────────
# Model Registry Management (MLflow Registered Models)
# ──────────────────────────────────────────────────────────────────────────────


@router.post("/registry", status_code=status.HTTP_201_CREATED)
def create_registered_model(
    name: str = Form(...),
    description: str = Form(default=""),
    _user: User = Depends(require_editor),
    tenant: Tenant = Depends(get_current_tenant),
):
    """
    Crea un nuevo modelo registrado en MLflow Model Registry.
    Este es el "contenedor" donde se almacenarán las versiones del modelo.
    Requiere rol editor o superior.
    """
    try:
        mlflow_svc = MLFlowService()
        model_name = f"tenant_{tenant.id}_{name.replace(' ', '_')}"
        result = mlflow_svc.create_registered_model(
            name=model_name,
            description=description,
        )
        return {
            "message": "Registered model created successfully",
            "name": result["name"],
            "description": result.get("description", ""),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating registered model: {exc}",
        )


@router.get("/registry", response_model=dict)
def list_registered_models(
    _user: User = Depends(require_viewer),
    tenant: Tenant = Depends(get_current_tenant),
):
    """
    Lista todos los modelos registrados en MLflow Model Registry para este tenant.
    """
    try:
        mlflow_svc = MLFlowService()
        all_models = mlflow_svc.get_registered_models()
        # Filter to only tenant's models
        tenant_prefix = f"tenant_{tenant.id}_"
        tenant_models = []
        for m in all_models["models"]:
            if m["name"].startswith(tenant_prefix):
                # Add display_name and keep original name for API calls
                m["display_name"] = m["name"][len(tenant_prefix) :]
                tenant_models.append(m)
            elif m["name"].startswith("global_"):
                m["display_name"] = m["name"]
                tenant_models.append(m)

        return {"models": tenant_models}
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing registered models: {exc}",
        )


@router.delete("/registry/{registry_name}")
def delete_registered_model(
    registry_name: str,
    _user: User = Depends(require_admin),
    tenant: Tenant = Depends(get_current_tenant),
):
    """
    Elimina un modelo registrado de MLflow Model Registry.
    Requiere rol admin.
    """
    try:
        mlflow_svc = MLFlowService()
        mlflow_svc.delete_registered_model(registry_name)
        return {"message": f"Registered model {registry_name} deleted"}
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting registered model: {exc}",
        )


@router.get("/registry/{model_name}/versions")
def get_registry_model_versions(
    model_name: str,
    _user: User = Depends(require_viewer),
):
    """
    Lista todas las versiones de un modelo registrado en MLflow.
    """
    try:
        mlflow_svc = MLFlowService()
        versions = mlflow_svc.get_model_versions(model_name)
        return {"versions": versions}
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching versions: {exc}",
        )


@router.get("/runs/{run_id}/details")
def get_run_details(
    run_id: str,
    _user: User = Depends(require_viewer),
):
    """
    Obtiene los detalles de un run (métricas, parámetros) de MLflow.
    """
    try:
        mlflow_svc = MLFlowService()
        details = mlflow_svc.get_run_details(run_id)
        return details
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching run details: {exc}",
        )


# ──────────────────────────────────────────────────────────────────────────────
# Model Registry - Versioning and Stage Management
# ──────────────────────────────────────────────────────────────────────────────


@router.post("/{model_id}/promote", response_model=MLModelResponse)
def promote_model(
    model_id: str,
    target_stage: str = Form(default="Production"),
    _user: User = Depends(require_editor),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """
    Promote a model to Production or archive it.
    Requires editor role or higher.
    """
    model = (
        db.query(MLModel)
        .filter(MLModel.id == model_id, MLModel.tenant_id == tenant.id)
        .first()
    )

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    if target_stage not in ["Staging", "Production", "Archived"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid stage. Must be 'Staging', 'Production', or 'Archived'",
        )

    if model.mlflow_run_id and model.mlflow_registry_name:
        try:
            mlflow_svc = MLFlowService()
            version = int(model.mlflow_version) if model.mlflow_version else 1
            mlflow_svc.transition_model_stage(
                model_name=model.mlflow_registry_name,
                version=version,
                stage=target_stage,
            )
        except Exception as e:
            logger.warning("MLflow registry transition failed: %s", str(e))

    model.stage = target_stage
    model.promoted_at = datetime.now(timezone.utc)
    model.promoted_by = str(_user.id)

    db.commit()
    db.refresh(model)
    return model


@router.post("/{model_id}/archive", response_model=MLModelResponse)
def archive_model(
    model_id: str,
    _user: User = Depends(require_editor),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """Archive a model (move to Archived stage)."""
    model = (
        db.query(MLModel)
        .filter(MLModel.id == model_id, MLModel.tenant_id == tenant.id)
        .first()
    )

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    if model.mlflow_run_id and model.mlflow_registry_name:
        try:
            mlflow_svc = MLFlowService()
            version = int(model.mlflow_version) if model.mlflow_version else 1
            mlflow_svc.transition_model_stage(
                model_name=model.mlflow_registry_name,
                version=version,
                stage="Archived",
            )
        except Exception as e:
            logger.warning("MLflow registry transition failed: %s", str(e))

    model.stage = "Archived"
    model.promoted_at = datetime.now(timezone.utc)
    model.promoted_by = str(_user.id)

    db.commit()
    db.refresh(model)
    return model


@router.get("/{model_id}/versions")
def get_model_versions(
    model_id: str,
    _user: User = Depends(require_viewer),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """Get version history for a model."""
    model = (
        db.query(MLModel)
        .filter(MLModel.id == model_id, MLModel.tenant_id == tenant.id)
        .first()
    )

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    versions = []
    if model.mlflow_registry_name:
        try:
            mlflow_svc = MLFlowService()
            versions = mlflow_svc.get_model_versions(model.mlflow_registry_name)
        except Exception as e:
            logger.warning("Failed to get MLflow versions: %s", str(e))

    return {
        "current_version": model.version,
        "stage": model.stage,
        "promoted_at": model.promoted_at,
        "promoted_by": model.promoted_by,
        "mlflow_versions": versions,
    }
