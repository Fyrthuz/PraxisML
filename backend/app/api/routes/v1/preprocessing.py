"""
API route para configurar y aplicar pipelines de preprocesamiento en datasets tabulares.
"""

import logging
import os
import tempfile
from io import BytesIO
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import (
    get_current_tenant,
    get_storage_service,
    require_editor,
    require_viewer,
)
from app.core_ml.preprocessing import (
    apply_pipeline,
    build_pipeline,
    save_pipeline,
)
from app.core_ml.tabular_parser import is_tabular, read_tabular
from app.database import get_db
from app.models.dataset import Dataset
from app.models.tenant import Tenant
from app.models.user import User
from app.services.storage_service import StorageService

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Schemas ──────────────────────────────────────────────────────────────────


class PreprocessingStep(BaseModel):
    type: str  # "impute", "scale", "encode", "feature_eng", "drop"
    columns: List[str]
    strategy: Optional[str] = None  # para impute: mean, median, most_frequent, constant
    method: Optional[str] = (
        None  # para scale/encode/feature_eng: standard, minmax, robust / onehot, ordinal / log_transform, polynomial, binning
    )
    fill_value: Optional[str] = None  # para impute strategy=constant
    bins: Optional[int] = None  # para feature_eng binning


class PreprocessingConfig(BaseModel):
    dataset_id: str
    target_column: Optional[str] = None
    steps: List[PreprocessingStep]


class PreprocessingPreviewResponse(BaseModel):
    original_columns: List[str]
    transformed_columns: List[str]
    original_shape: List[int]  # [rows, cols]
    transformed_shape: List[int]  # [rows, cols]
    preview_rows: List[dict]  # First 10 transformed rows
    pipeline_path: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────────
# POST /preprocessing/preview  — Configurar y previsualizar
# ──────────────────────────────────────────────────────────────────────────────
@router.post("/preview", response_model=PreprocessingPreviewResponse)
def preview_preprocessing(
    config: PreprocessingConfig,
    _user: User = Depends(require_editor),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
    storage: StorageService = Depends(get_storage_service),
):
    """
    Recibe la configuración de preprocesamiento, la aplica al dataset,
    y devuelve un preview de las primeras 10 filas transformadas.
    No persiste nada — solo validación y previsualización.

    Requiere rol **editor** o superior.
    """
    dataset = _get_tabular_dataset(config.dataset_id, tenant, db)

    try:
        data_bytes = storage.download(dataset.file_path)
        df = read_tabular(BytesIO(data_bytes), dataset.file_type)
    except Exception as e:
        logger.error(f"Error downloading dataset for preview: {e}")
        raise HTTPException(status_code=500, detail=f"Error downloading dataset from storage: {e}")

    try:
        pipeline_config = {"steps": [s.model_dump() for s in config.steps]}
        pipeline = build_pipeline(pipeline_config, df.columns.tolist())
        df_transformed, y = apply_pipeline(pipeline, df, config.target_column)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en preprocesamiento: {e}")

    preview = (
        df_transformed.head(10)
        .where(df_transformed.head(10).notna(), None)
        .to_dict(orient="records")
    )

    return PreprocessingPreviewResponse(
        original_columns=df.columns.tolist(),
        transformed_columns=df_transformed.columns.tolist(),
        original_shape=[len(df), len(df.columns)],
        transformed_shape=[len(df_transformed), len(df_transformed.columns)],
        preview_rows=preview,
    )


# ──────────────────────────────────────────────────────────────────────────────
# POST /preprocessing/apply/{dataset_id}  — Aplicar y guardar
# ──────────────────────────────────────────────────────────────────────────────
@router.post("/apply/{dataset_id}", response_model=dict)
def apply_preprocessing(
    dataset_id: str,
    config: PreprocessingConfig,
    _user: User = Depends(require_editor),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
    storage: StorageService = Depends(get_storage_service),
):
    """
    Aplica el pipeline de preprocesamiento al dataset, guarda:
      1. El dataset transformado como CSV (nueva versión del dataset)
      2. El pipeline serializado como artefacto (.joblib) para inferencia

    Requiere rol **editor** o superior.

    Returns:
        dict con el nuevo dataset_id y la ruta del pipeline guardado.
    """
    dataset = _get_tabular_dataset(dataset_id, tenant, db)

    try:
        data_bytes = storage.download(dataset.file_path)
        df = read_tabular(BytesIO(data_bytes), dataset.file_type)
    except Exception as e:
        logger.error(f"Error downloading dataset for preprocessing: {e}")
        raise HTTPException(status_code=500, detail=f"Error downloading dataset from storage: {e}")

    try:
        pipeline_config = {
            "steps": [s.model_dump() for s in config.steps],
            "target_column": config.target_column,
        }
        pipeline = build_pipeline(pipeline_config, df.columns.tolist())
        df_transformed, y = apply_pipeline(pipeline, df, config.target_column)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # ── Guardar pipeline en MLFlow ─────────────────────────────────────────
    pipeline_name = f"{dataset.name}_pipeline"
    pipeline_path = save_pipeline(pipeline, pipeline_name, tenant.id, pipeline_config)

    # ── Calcular nueva versión ──────────────────────────────────────────────
    existing_count = (
        db.query(Dataset)
        .filter(Dataset.tenant_id == tenant.id, Dataset.name == dataset.name)
        .count()
    )
    new_version = existing_count + 1

    # Si hay target column, reincorporarla
    if y is not None and config.target_column:
        df_transformed[config.target_column] = y.values

    new_filename = f"{dataset.name}_v{new_version}_preprocessed.csv"
    storage_key = f"tenants/{tenant.id}/datasets/{new_filename}"

    # ── Guardar temporalmente para DVC y extracción de metadata ──────────────
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            df_transformed.to_csv(tmp.name, index=False)
            tmp_path = tmp.name

        # Upload to storage
        with open(tmp_path, "rb") as f:
            content = f.read()
            storage.upload(storage_key, content)

        file_size = len(content)
        new_file_path = storage_key # La ruta en BD será la storage key

        # ── Registrar en DVC (opcional si está configurado) ──────────────────
        from app.services.dvc_service import DVCService, track_dataset_with_dvc

        registry_name = dataset.dvc_registry_name or dataset.name
        dvc_info = {}
        try:
            dvc_service = DVCService(tenant.id)
            # DVC requiere que el archivo esté dentro de su workspace (repo)
            local_path = dvc_service.get_local_path(storage_key)

            # Aseguramos que el archivo existe en el workspace local antes de trackear
            if not local_path.exists():
                local_path.parent.mkdir(parents=True, exist_ok=True)
                with open(local_path, "wb") as f:
                    f.write(content)

            dvc_info = track_dataset_with_dvc(
                tenant_id=tenant.id,
                file_path=str(local_path),
                registry_name=registry_name,
            )
            logger.info(f"Dataset preprocesado registrado en DVC: {storage_key}")
        except Exception as e:
            logger.warning(f"Error al trackear con DVC: {e}")

        # ── Registrar nuevo dataset en BD ────────────────────────────────────
        new_dataset = Dataset(
            name=f"{dataset.name} (v{new_version} preprocessed)",
            description=f"Preprocesado desde '{dataset.name}' (v{dataset.version}). Target: {config.target_column or 'N/A'}",
            file_path=new_file_path,
            file_size_bytes=file_size,
            file_type="csv",
            num_rows=len(df_transformed),
            num_columns=len(df_transformed.columns),
            column_names=df_transformed.columns.tolist(),
            version=new_version,
            pipeline_path=pipeline_path,
            tenant_id=tenant.id,
            # Campos DVC
            is_dvc_tracked=dvc_info.get("is_dvc_tracked", False),
            dvc_hash=dvc_info.get("dvc_hash"),
            dvc_remote=dvc_info.get("dvc_remote"),
            dvc_registry_name=dvc_info.get("dvc_registry_name"),
            dvc_version=dvc_info.get("dvc_version"),
        )
        db.add(new_dataset)
        db.commit()
        db.refresh(new_dataset)

        return {
            "message": "Preprocesamiento aplicado correctamente.",
            "new_dataset_id": new_dataset.id,
            "new_dataset_name": new_dataset.name,
            "pipeline_path": pipeline_path,
            "transformed_shape": [len(df_transformed), len(df_transformed.columns)],
        }
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# ──────────────────────────────────────────────────────────────────────────────
# GET /preprocessing/pipeline/{dataset_id}  — Obtener configuración del pipeline
# ──────────────────────────────────────────────────────────────────────────────
@router.get("/pipeline/{dataset_id}")
def get_dataset_pipeline_config(
    dataset_id: str,
    _user: User = Depends(require_viewer),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """
    Recupera la configuración (pasos) del pipeline asociado a un dataset.
    Busca en los tags/params del run de MLFlow si existe.
    Requiere rol **viewer** o superior.
    """
    dataset = (
        db.query(Dataset)
        .filter(Dataset.id == dataset_id, Dataset.tenant_id == tenant.id)
        .first()
    )

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset no encontrado.")

    if not dataset.pipeline_path:
        return {
            "steps": [],
            "message": "Este dataset no tiene un pipeline de preprocesamiento asociado.",
        }

    # Si el pipeline_path es una URI de MLFlow (runs:/...), podemos intentar sacar info
    if dataset.pipeline_path.startswith("runs:/"):
        try:
            from mlflow.tracking import MlflowClient

            from app.services.mlflow_service import MLFlowService

            run_id = dataset.pipeline_path.split("/")[1]
            mlflow_svc = MLFlowService()
            client = MlflowClient(tracking_uri=mlflow_svc.tracking_uri)
            run = client.get_run(run_id)

            # Buscamos en los tags si guardamos la config allí
            import json

            steps_json = run.data.tags.get("pipeline_steps")
            if steps_json:
                return {
                    "steps": json.loads(steps_json),
                    "run_id": run_id,
                    "target_column": run.data.tags.get("target_column"),
                }
        except Exception as e:
            logger.warning(f"No se pudo recuperar info detallada de MLFlow: {e}")

    return {
        "steps": [],
        "pipeline_path": dataset.pipeline_path,
        "message": "Pipeline encontrado pero la configuración detallada no está disponible en MLFlow.",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _get_tabular_dataset(dataset_id: str, tenant: Tenant, db: Session) -> Dataset:
    """Obtiene un dataset validando que sea tabular y pertenezca al tenant."""
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
            detail=f"Preprocesamiento solo disponible para datasets tabulares. Tipo actual: {dataset.file_type}",
        )
    return dataset
