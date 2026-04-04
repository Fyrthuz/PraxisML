"""
Endpoints para Data Drift
"""

import logging
import os
from datetime import datetime
from io import BytesIO
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_current_tenant, require_viewer
from app.database import get_db
from app.models.dataset import Dataset
from app.models.ml_model import MLModel
from app.models.tenant import Tenant
from app.models.user import User
from app.services.storage_service import get_storage

# Evidently imports
try:
    from evidently import ColumnMapping
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False

router = APIRouter()

logger = logging.getLogger(__name__)


@router.get("/drift/report/dataset/{dataset_id}")
def get_dataset_drift_report(
    dataset_id: str,
    db: Session = Depends(get_db),
    current_tenant: Tenant = Depends(get_current_tenant),
    _: User = Depends(require_viewer),
):
    """
    Genera un reporte de drift para un dataset específico.
    Compara el dataset actual con una referencia (ejemplo: versión anterior).
    """
    logger.info(
        f"Getting drift report for dataset {dataset_id} in tenant {current_tenant.id}"
    )

    if not EVIDENTLY_AVAILABLE:
        logger.warning("Evidently no está instalado, devolviendo datos simulados")
        # Devolver datos simulados si Evidently no está disponible
        return {
            "dataset_id": dataset_id,
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "dataset_drift": False,
                "drift_by_columns": {},
                "psi_threshold": 0.2,
                "ks_threshold": 0.05,
            },
            "simulated": True,
        }

    # Obtener dataset
    dataset = (
        db.query(Dataset)
        .filter(Dataset.id == dataset_id, Dataset.tenant_id == current_tenant.id)
        .first()
    )

    if not dataset:
        logger.error(f"Dataset {dataset_id} not found for tenant {current_tenant.id}")
        raise HTTPException(status_code=404, detail="Dataset no encontrado")

    # Calcular drift real usando Evidently
    try:
        # Descargar dataset actual desde MinIO/S3/Local
        storage = get_storage()
        logger.info(f"Descargando dataset desde storage: {dataset.file_path}")

        try:
            # Descargar como bytes y cargar en pandas
            data_bytes = storage.download(dataset.file_path)
            # Usar StringIO para leer el CSV desde bytes
            from io import BytesIO

            df_current = pd.read_csv(BytesIO(data_bytes))
        except Exception as e:
            logger.error(f"Error descargando dataset: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error descargando dataset desde storage: {str(e)}",
            )

        # Obtener umbrales de drift del metadata del dataset
        metadata = dataset.metrics_metadata or {}
        psi_threshold = metadata.get("drift_psi_threshold", 0.2)
        ks_threshold = metadata.get("drift_ks_threshold", 0.05)

        # Determinar dataset de referencia
        # 1. Intentar cargar dataset de referencia desde DVC (versión anterior)
        df_reference = None

        if dataset.parent_dataset_id:
            # Cargar dataset padre si existe
            parent_dataset = (
                db.query(Dataset)
                .filter(
                    Dataset.id == dataset.parent_dataset_id,
                    Dataset.tenant_id == current_tenant.id,
                )
                .first()
            )
            if parent_dataset and parent_dataset.file_path:
                try:
                    logger.info(
                        f"Descargando dataset de referencia desde: {parent_dataset.file_path}"
                    )
                    parent_data_bytes = storage.download(parent_dataset.file_path)
                    df_reference = pd.read_csv(BytesIO(parent_data_bytes))
                except Exception as e:
                    logger.warning(f"No se pudo cargar dataset padre: {e}")

        # 2. Si no hay dataset padre, usar muestra del mismo dataset como referencia
        if df_reference is None:
            logger.info("Usando muestra del dataset actual como referencia")
            df_reference = df_current.sample(frac=0.8, random_state=42)
            df_current = df_current.drop(df_reference.index)

        # Configurar column mapping (asumimos que todas las columnas son features)
        column_mapping = ColumnMapping()

        # Identificar columnas categóricas (si las hay)
        categorical_columns = df_current.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        if categorical_columns:
            column_mapping.categorical_features = categorical_columns

        # Identificar columnas numéricas
        numerical_columns = df_current.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        if numerical_columns:
            column_mapping.numerical_features = numerical_columns

        # Generar reporte de drift
        logger.info("Generando reporte de drift con Evidently")
        report = Report(metrics=[DataDriftPreset()])
        report.run(
            reference_data=df_reference,
            current_data=df_current,
            column_mapping=column_mapping,
        )

        # Extraer resultados
        result_dict = report.as_dict()

        logger.info(f"Estructura de resultado Evidently: {list(result_dict.keys())}")

        # Procesar resultados para el formato esperado por el frontend
        # Intentar diferentes estructuras de datos de Evidently
        drift_detected = False
        drift_by_columns = {}

        try:
            # Estructura típica de Evidently
            if "metrics" in result_dict:
                for metric_name, metric_data in result_dict["metrics"].items():
                    if (
                        "DataDrift" in metric_name
                        or "data_drift" in metric_name.lower()
                    ):
                        if "result" in metric_data:
                            result_data = metric_data["result"]

                            # Drift detectado a nivel de dataset
                            if "dataset_drift" in result_data:
                                drift_detected = result_data["dataset_drift"]

                            # Drift por columnas
                            if "drift_by_columns" in result_data:
                                drift_by_columns = result_data["drift_by_columns"]
                            elif "drifted_columns" in result_data:
                                for col in result_data["drifted_columns"]:
                                    drift_by_columns[col] = {"drift_detected": True}

                            # Alternativa: column_drift
                            if "column_drift" in result_data:
                                for col_name, col_data in result_data[
                                    "column_drift"
                                ].items():
                                    drift_by_columns[col_name] = {
                                        "drift_detected": col_data.get(
                                            "drift_detected", False
                                        ),
                                        "psi": col_data.get("psi", None),
                                        "ks": col_data.get("ks", None),
                                    }

                            break

            # Si no encontramos datos de drift, intentar con columnas separadas
            if not drift_by_columns and "metrics" in result_dict:
                for metric_name, metric_data in result_dict["metrics"].items():
                    if "ColumnDrift" in metric_name:
                        # Extraer nombre de columna del nombre de la métrica
                        # Formato típico: "ColumnDrift for column_name"
                        col_name = metric_name.replace("ColumnDrift for ", "").replace(
                            "ColumnDrift_", ""
                        )
                        if "result" in metric_data:
                            col_result = metric_data["result"]
                            drift_by_columns[col_name] = {
                                "drift_detected": col_result.get(
                                    "drift_detected", False
                                ),
                                "psi": col_result.get("psi_score", None),
                                "ks": col_result.get("ks_score", None),
                            }

        except Exception as e:
            logger.error(f"Error procesando resultados de Evidently: {e}")
            # En caso de error, usar valores por defecto
            drift_detected = False
            drift_by_columns = {}

        # Formatear resultado
        result = {
            "dataset_id": dataset_id,
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "dataset_drift": drift_detected,
                "drift_by_columns": drift_by_columns,
                "psi_threshold": psi_threshold,
                "ks_threshold": ks_threshold,
            },
            "reference_info": {
                "reference_rows": len(df_reference),
                "current_rows": len(df_current),
            },
        }

        logger.info(
            f"Drift calculado. Drift detectado: {drift_detected}, Columnas con drift: {list(drift_by_columns.keys())}"
        )

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
    current_tenant: Tenant = Depends(get_current_tenant),
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
        .filter(MLModel.id == model_id, MLModel.tenant_id == current_tenant.id)
        .first()
    )

    if not model:
        raise HTTPException(status_code=404, detail="Modelo no encontrado")

    try:
        from app.models.prediction import Prediction as PredictionModel

        storage = get_storage()
        metadata = model.metrics_metadata or {}

        # 1. Obtener dataset de referencia (entrenamiento)
        df_reference = None
        dataset_id = metadata.get("dataset_id")

        if dataset_id:
            dataset = db.query(Dataset).filter(
                Dataset.id == dataset_id,
                Dataset.tenant_id == current_tenant.id
            ).first()

            if dataset and dataset.file_path:
                try:
                    logger.info(f"Cargando dataset de entrenamiento: {dataset.file_path}")
                    ref_bytes = storage.download(dataset.file_path)
                    df_reference = pd.read_csv(BytesIO(ref_bytes))

                    # Si el dataset tiene pipeline, aplicarlo para tener los datos post-procesados
                    if model.preprocessing_pipeline_path:
                        from app.core_ml.preprocessing import (
                            apply_pipeline,
                            load_pipeline,
                        )
                        try:
                            pipeline = load_pipeline(model.preprocessing_pipeline_path)
                            df_reference, _ = apply_pipeline(pipeline, df_reference)
                        except Exception as e:
                            logger.warning(f"No se pudo aplicar pipeline al dataset de referencia: {e}")
                except Exception as e:
                    logger.warning(f"No se pudo cargar dataset de entrenamiento: {e}")

        if df_reference is None:
            raise HTTPException(
                status_code=400,
                detail="No se encontró dataset de referencia para este modelo"
            )

        # 2. Recolectar datos de inferencia recientes
        # Buscamos las últimas 200 predicciones exitosas
        predictions = (
            db.query(PredictionModel)
            .filter(
                PredictionModel.model_id == model_id,
                PredictionModel.tenant_id == current_tenant.id,
                PredictionModel.status == "COMPLETED",
                PredictionModel.input_image_path.isnot(None)
            )
            .order_by(PredictionModel.created_at.desc())
            .limit(200)
            .all()
        )

        if not predictions:
            raise HTTPException(
                status_code=400,
                detail="No hay datos de inferencia suficientes para calcular drift"
            )

        inference_data_list = []
        for p in predictions:
            try:
                # Los datos de input se guardan como .npy en disco (según single_predict.py)
                if os.path.exists(p.input_image_path):
                    arr = np.load(p.input_image_path)
                    inference_data_list.append(arr.flatten())
            except Exception as e:
                logger.warning(f"Error cargando datos de predicción {p.id}: {e}")

        if not inference_data_list:
            raise HTTPException(
                status_code=400,
                detail="No se pudieron cargar los datos de inferencia desde disco"
            )

        # Reconstruir DataFrame de inferencia
        feature_names = metadata.get("feature_names", [])
        if not feature_names:
            feature_names = [f"feature_{i}" for i in range(len(inference_data_list[0]))]

        df_current = pd.DataFrame(inference_data_list, columns=feature_names)

        # 3. Calcular Drift con Evidently
        column_mapping = ColumnMapping()
        # Intentar alinear columnas si es necesario (df_reference puede tener target, df_current no)
        common_cols = [c for c in feature_names if c in df_reference.columns]
        df_reference_subset = df_reference[common_cols]
        df_current_subset = df_current[common_cols]

        report = Report(metrics=[DataDriftPreset()])
        report.run(
            reference_data=df_reference_subset,
            current_data=df_current_subset,
            column_mapping=column_mapping
        )

        result_dict = report.as_dict()

        # Extraer métricas resumen
        drift_detected = False
        drift_by_columns = {}

        try:
            for metric in result_dict.get("metrics", []):
                if metric.get("metric") == "DatasetDriftMetric":
                    drift_detected = metric.get("result", {}).get("dataset_drift", False)
                elif metric.get("metric") == "DataDriftTable":
                    drift_by_columns = metric.get("result", {}).get("drift_by_columns", {})
        except Exception as e:
            logger.error(f"Error parseando resultado Evidently: {e}")

        return {
            "model_id": model_id,
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "dataset_drift": drift_detected,
                "drift_by_columns": drift_by_columns,
                "psi_threshold": metadata.get("drift_psi_threshold", 0.2),
                "ks_threshold": metadata.get("drift_ks_threshold", 0.05),
            },
            "reference_info": {
                "reference_rows": len(df_reference_subset),
                "current_rows": len(df_current_subset),
                "num_features": len(common_cols)
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generando reporte de drift del modelo: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error generando reporte: {str(e)}"
        )


@router.patch("/datasets/{dataset_id}/drift-thresholds")
def update_dataset_drift_thresholds(
    dataset_id: str,
    psi_threshold: Optional[float] = None,
    ks_threshold: Optional[float] = None,
    db: Session = Depends(get_db),
    current_tenant: Tenant = Depends(get_current_tenant),
    _: User = Depends(require_viewer),
):
    """
    Actualiza los umbrales de drift para un dataset específico.
    """
    dataset = (
        db.query(Dataset)
        .filter(Dataset.id == dataset_id, Dataset.tenant_id == current_tenant.id)
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
    current_tenant: Tenant = Depends(get_current_tenant),
    _: User = Depends(require_viewer),
):
    """
    Actualiza los umbrales de drift para un modelo específico.
    """
    model = (
        db.query(MLModel)
        .filter(MLModel.id == model_id, MLModel.tenant_id == current_tenant.id)
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
