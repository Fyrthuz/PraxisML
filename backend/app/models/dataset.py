from sqlalchemy import Column, String, DateTime, ForeignKey, Integer, JSON, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from app.models.base import Base
import uuid


def generate_uuid():
    return str(uuid.uuid4())


class Dataset(Base):
    id = Column(String, primary_key=True, default=generate_uuid, index=True)
    name = Column(String, nullable=False)
    description = Column(String, nullable=True)
    file_path = Column(String, nullable=False)  # Ruta local donde está el archivo
    config_path = Column(String, nullable=True)  # Ruta local donde está el config.json
    file_size_bytes = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # ── Nuevos campos Fase 1: Data Ops ────────────────────────────────────────
    file_type = Column(String, nullable=True)  # csv, xlsx, parquet, zip
    num_rows = Column(Integer, nullable=True)  # Nº filas (solo tabular)
    num_columns = Column(Integer, nullable=True)  # Nº columnas (solo tabular)
    column_names = Column(JSON, nullable=True)  # Lista de nombres de columnas
    version = Column(Integer, default=1)  # Versión del dataset
    mlflow_artifact_uri = Column(String, nullable=True)  # URI en MLFlow Artifacts
    pipeline_path = Column(
        String, nullable=True
    )  # Ruta local a .joblib si fue preprocesado

    # ── DVC Versioning ───────────────────────────────────────────────────────
    dvc_remote = Column(String, nullable=True)  # Remote name (minio, awss3)
    dvc_hash = Column(String, nullable=True)  # MD5 hash actual del archivo
    is_dvc_tracked = Column(Boolean, default=False)  # Si está trackeado con DVC
    dvc_registry_name = Column(String, nullable=True)  # Nombre del registry en DVC
    dvc_version = Column(Integer, nullable=True)  # Versión en DVC (1, 2, 3...)
    parent_dataset_id = Column(
        String, nullable=True
    )  # Dataset original si es versión derivada
    is_active = Column(Boolean, default=False)  # Si es la versión activa/prod en el registry

    # Relación con el Tenant (Aislamiento Multi-Tenant)
    tenant_id = Column(String, ForeignKey("tenant.id"), nullable=False, index=True)
    tenant = relationship("Tenant", back_populates="datasets")

    # Un dataset puede generar múltiples predicciones
    # predictions = relationship("Prediction", back_populates="dataset")
