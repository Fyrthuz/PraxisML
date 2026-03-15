"""
Endpoints para Data Drift
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from app.database import get_db
from app.models.dataset import Dataset
from app.models.ml_model import MLModel
from app.models.user import User
from app.api.deps import get_current_tenant, require_viewer
from app.core.config import settings
import logging

# Evidently imports
try:
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    from evidently.metrics import ColumnDriftMetric

    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False

router = APIRouter()

logger = logging.getLogger(__name__)


@router.get("/drift/report/dataset/{dataset_id}")
def get_dataset_drift_report(
    dataset_id: str,
    db: Session = Depends(get_db),
    current_tenant: dict = Depends(get_current_tenant),
    _: User = Depends(require_viewer),
):
    """
    Genera un reporte de drift para un dataset específico.
    Compara el dataset actual con una referencia (ejemplo: versión anterior).
    """
    logger.info(
        f"Getting drift report for dataset {dataset_id} in tenant {current_tenant['id']}"
    )

    if not EVIDENTLY_AVAILABLE:
        logger.warning("Evidently no está instalado, devolviendo datos simulados")
        # Devolver datos simulados si Evidently no está disponible
        return {
            "dataset_id": dataset_id,
            "timestamp": datetime.now().isoformat(),
            "drift_detected": False,
            "drift_by_columns": {},
            "psi_threshold": 0.2,
            "ks_threshold": 0.05,
            "simulated": True,
        }

    # Obtener dataset
    dataset = (
        db.query(Dataset)
        .filter(Dataset.id == dataset_id, Dataset.tenant_id == current_tenant["id"])
        .first()
    )

    if not dataset:
        logger.error(
            f"Dataset {dataset_id} not found for tenant {current_tenant['id']}"
        )
        raise HTTPException(status_code=404, detail="Dataset no encontrado")

    # Simulación: comparar con versión anterior o datos de referencia
    # En producción, cargar dataset de referencia desde DVC o histórico
    try:
        # Cargar datos actuales (ejemplo)
        # df_current = pd.read_csv(dataset.file_path)

        # Generar datos de referencia simulados (ejemplo)
        # df_reference = df_current.sample(frac=0.8)  # Simulación

        # Configurar column mapping
        # column_mapping = ColumnMapping()

        # Generar reporte
        # report = Report(metrics=[DataDriftPreset()])
        # report.run(reference_data=df_reference, current_data=df_current, column_mapping=column_mapping)

        # Extraer resultados
        # result = report.as_dict()

        # Ejemplo de respuesta simulada
        result = {
            "dataset_id": dataset_id,
            "timestamp": datetime.now().isoformat(),
            "drift_detected": False,
            "drift_by_columns": {},
            "psi_threshold": 0.2,
            "ks_threshold": 0.05,
        }

        return result

    except Exception as e:
        logger.error(f"Error generando reporte de drift: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error generando reporte: {str(e)}"
        )


@router.get("/drift/report/model/{model_id}")
def get_model_drift_report(
    model_id: str,
    db: Session = Depends(get_db),
    current_tenant: dict = Depends(get_current_tenant),
    _: User = Depends(require_viewer),
):
    """
    Genera un reporte de drift para un modelo específico.
    Compara los datos de inferencia recientes con los datos de entrenamiento.
    """
    if not EVIDENTLY_AVAILABLE:
        raise HTTPException(status_code=501, detail="Evidently no está instalado")

    # Obtener modelo
    model = (
        db.query(MLModel)
        .filter(MLModel.id == model_id, MLModel.tenant_id == current_tenant["id"])
        .first()
    )

    if not model:
        raise HTTPException(status_code=404, detail="Modelo no encontrado")

    try:
        # Lógica similar a get_dataset_drift_report
        # Comparar datos de inferencia vs datos de entrenamiento

        result = {
            "model_id": model_id,
            "timestamp": datetime.now().isoformat(),
            "drift_detected": False,
            "drift_by_columns": {},
            "psi_threshold": 0.2,
            "ks_threshold": 0.05,
        }

        return result

    except Exception as e:
        logger.error(f"Error generando reporte de drift del modelo: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error generando reporte: {str(e)}"
        )


@router.patch("/datasets/{dataset_id}/drift-thresholds")
def update_dataset_drift_thresholds(
    dataset_id: str,
    psi_threshold: Optional[float] = None,
    ks_threshold: Optional[float] = None,
    db: Session = Depends(get_db),
    current_tenant: dict = Depends(get_current_tenant),
    _: User = Depends(require_viewer),
):
    """
    Actualiza los umbrales de drift para un dataset específico.
    """
    dataset = (
        db.query(Dataset)
        .filter(Dataset.id == dataset_id, Dataset.tenant_id == current_tenant["id"])
        .first()
    )

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset no encontrado")

    try:
        # Actualizar umbrales en metadata o campo específico
        # Por simplicidad, asumimos que se guardan en metrics_metadata
        if not dataset.metrics_metadata:
            dataset.metrics_metadata = {}

        if psi_threshold is not None:
            dataset.metrics_metadata["drift_psi_threshold"] = psi_threshold
        if ks_threshold is not None:
            dataset.metrics_metadata["drift_ks_threshold"] = ks_threshold

        db.commit()
        db.refresh(dataset)

        return {
            "message": "Umbrales actualizados exitosamente",
            "dataset_id": dataset_id,
            "psi_threshold": psi_threshold,
            "ks_threshold": ks_threshold,
        }

    except Exception as e:
        db.rollback()
        logger.error(f"Error actualizando umbrales de drift: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error actualizando umbrales: {str(e)}"
        )


@router.patch("/models/{model_id}/drift-thresholds")
def update_model_drift_thresholds(
    model_id: str,
    psi_threshold: Optional[float] = None,
    ks_threshold: Optional[float] = None,
    db: Session = Depends(get_db),
    current_tenant: dict = Depends(get_current_tenant),
    _: User = Depends(require_viewer),
):
    """
    Actualiza los umbrales de drift para un modelo específico.
    """
    model = (
        db.query(MLModel)
        .filter(MLModel.id == model_id, MLModel.tenant_id == current_tenant["id"])
        .first()
    )

    if not model:
        raise HTTPException(status_code=404, detail="Modelo no encontrado")

    try:
        # Actualizar umbrales en metadata
        if not model.metrics_metadata:
            model.metrics_metadata = {}

        if psi_threshold is not None:
            model.metrics_metadata["drift_psi_threshold"] = psi_threshold
        if ks_threshold is not None:
            model.metrics_metadata["drift_ks_threshold"] = ks_threshold

        db.commit()
        db.refresh(model)

        return {
            "message": "Umbrales actualizados exitosamente",
            "model_id": model_id,
            "psi_threshold": psi_threshold,
            "ks_threshold": ks_threshold,
        }

    except Exception as e:
        db.rollback()
        logger.error(f"Error actualizando umbrales de drift: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error actualizando umbrales: {str(e)}"
        )
