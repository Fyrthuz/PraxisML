import shutil
import mlflow
import os
import tempfile
import zipfile
import csv
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.ml_model import MLModel
from app.models.user import User
from app.models.tenant import Tenant
from app.schemas.ml_model import MLModelCreate, MLModelResponse
from app.api.deps import (
    get_current_tenant,
    require_editor,
    require_viewer,
    require_admin,
    check_model_quota,
)
from app.services.mlflow_service import MLFlowService
from app.core.config import settings

router = APIRouter()


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

    existing = db.query(MLModel).filter(MLModel.mlflow_run_id == model_in.mlflow_run_id).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Este run_id de MLFlow ya está registrado en la base de datos.",
        )

    new_model = MLModel(**model_in.model_dump(), tenant_id=tenant.id)
    db.add(new_model)
    db.commit()
    db.refresh(new_model)
    return new_model


# ──────────────────────────────────────────────────────────────────────────────
# Upload .pth → register in MLFlow → save to DB  (el flujo integrado)
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/upload", response_model=MLModelResponse, status_code=status.HTTP_201_CREATED)
def upload_and_register_model(
    name: str = Form(...),
    description: Optional[str] = Form(None),
    architecture: str = Form("unknown"),
    num_classes: int = Form(2),
    is_public: bool = Form(False),
    file: UploadFile = File(...),
    _user: User = Depends(require_editor),
    tenant: Tenant = Depends(check_model_quota),
    db: Session = Depends(get_db),
):
    """
    Sube un fichero .pth, lo registra en MLFlow automáticamente y crea el
    registro en la base de datos.  Todo en un único paso.

    Requiere rol **editor** o superior. Valida cuota de modelos del tenant.

    Flujo:
      1. Guarda el .pth en DATA_DIR/tenants/<tenant_id>/models/
      2. Llama a MLFlowService.register_pth_model() → devuelve un run_id
      3. Crea MLModel en BD con ese run_id
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
        mlflow_run_id=run_id, # Será None si es TorchScript
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

@router.get("/", response_model=List[MLModelResponse])
def get_models(
    _user: User = Depends(require_viewer),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """
    Obtener modelos del tenant MÁS los modelos base disponibles globalmente.
    Requiere rol **viewer** o superior.
    """
    models = db.query(MLModel).filter(
        (MLModel.tenant_id == tenant.id) | (MLModel.is_public is True)
    ).all()
    return models


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
    model = db.query(MLModel).filter(
        MLModel.id == model_id,
        MLModel.tenant_id == tenant.id
    ).first()

    if not model:
        raise HTTPException(status_code=404, detail="Model no encontrado o sin permisos.")

    # Opcional: Eliminar archivos .pth/.pt de disco si existen
    if model.torchscript_path and os.path.exists(model.torchscript_path):
        try:
            os.remove(model.torchscript_path)
        except Exception as e:
            print(f"Warning: No se pudo borrar el archivo de torchscript {model.torchscript_path}: {e}")

    # Intentar eliminar de MLFlow
    if model.mlflow_run_id:
        try:
            mlflow_svc = MLFlowService()
            client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_svc.get_tracking_uri())
            client.delete_run(model.mlflow_run_id)
        except Exception as e:
            print(f"Warning: No se pudo borrar el run {model.mlflow_run_id} de MLFlow: {e}")

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
    model = db.query(MLModel).filter(
        MLModel.id == model_id,
        (MLModel.tenant_id == tenant_id) | (MLModel.is_public is True)
    ).first()

    if not model:
        raise HTTPException(status_code=404, detail="Model no encontrado.")

    # Crear directorio temporal para preparar el zip
    temp_dir = tempfile.mkdtemp(prefix="mlmodel_")

    try:
        # 1. Crear el CSV con metadatos
        csv_path = os.path.join(temp_dir, "model_info.csv")
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Key", "Value"])
            writer.writerow(["Model Name", model.name])
            writer.writerow(["Description", model.description or ""])
            writer.writerow(["MLFlow Run ID", model.mlflow_run_id or ""])
            writer.writerow(["Created At", str(model.created_at)])

            # Agregar metadatos e hiperparámetros
            if model.metrics_metadata:
                for k, v in model.metrics_metadata.items():
                    # Evitar listas largas como las features, se pueden serializar a json
                    if isinstance(v, (list, dict)):
                        import json
                        v = json.dumps(v)
                    writer.writerow([f"metadata_{k}", str(v)])

        # 2. Descargar artefactos de MLflow
        if model.mlflow_run_id:
            try:
                mlflow_svc = MLFlowService()
                client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_svc.get_tracking_uri())
                artifact_path = client.download_artifacts(run_id=model.mlflow_run_id, path="")

                # Copiar contenidos del directorio de artefactos
                dest_artifacts = os.path.join(temp_dir, "artifacts")
                shutil.copytree(artifact_path, dest_artifacts, dirs_exist_ok=True)
            except Exception as e:
                # Si falla, loguear pero continuar con el zip (puede que no haya artefactos)
                print(f"Warning: Podría no haberse descargado artefactos para {model.mlflow_run_id}: {e}")

        # Si el modelo original era TorchScript/Pth subido manual y tenemos path
        if model.is_torchscript and model.torchscript_path and os.path.exists(model.torchscript_path):
             shutil.copy2(model.torchscript_path, temp_dir)

        # 3. Descargar pipeline de preprocesamiento (si existe)
        if model.preprocessing_pipeline_path and model.preprocessing_pipeline_path.startswith("runs:/"):
            try:
                pipeline_run_id = model.preprocessing_pipeline_path.split("/")[1]
                pipeline_file_path = "/".join(model.preprocessing_pipeline_path.split("/")[2:])
                
                mlflow_svc = MLFlowService()
                client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_svc.get_tracking_uri())
                downloaded_pipe_path = client.download_artifacts(run_id=pipeline_run_id, path=pipeline_file_path)
                
                if os.path.isfile(downloaded_pipe_path):
                    shutil.copy2(downloaded_pipe_path, os.path.join(temp_dir, "preprocessing_pipeline.joblib"))
                elif os.path.isdir(downloaded_pipe_path):
                    shutil.copytree(downloaded_pipe_path, os.path.join(temp_dir, "preprocessing_pipeline"), dirs_exist_ok=True)
            except Exception as e:
                print(f"Warning: No se pudo descargar el pipeline de preprocesamiento: {e}")

        # 4. Crear archivo Zip
        zip_path = tempfile.mktemp(suffix=".zip", prefix=f"{model.name.replace(' ', '_')}_")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_dir)
                    zipf.write(file_path, arcname)

        return FileResponse(
            path=zip_path,
            filename=f"{model.name.replace(' ', '_')}.zip",
            media_type="application/zip",
            background=None
        )

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creando el paquete del modelo: {exc}"
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
