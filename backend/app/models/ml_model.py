from sqlalchemy import Column, String, DateTime, Boolean, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from app.models.base import Base
from enum import Enum as PyEnum
import uuid


class ModelStage(str, PyEnum):
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"


def generate_uuid():
    return str(uuid.uuid4())


class MLModel(Base):
    """
    Rastrea los modelos registrados en MLFlow que el usuario puede usar.

    Las columnas *_path guardan object keys del StorageService
    (e.g. "{tenant_id}/models/{id}/pipeline.joblib"), NO rutas de disco.
    Usa get_storage().download(key) para obtener el contenido.
    """

    id = Column(String, primary_key=True, default=generate_uuid, index=True)
    name = Column(String, nullable=False)
    description = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # ID del run en MLFlow para cargarlo vía mlflow.pytorch.load_model(f"runs:/{run_id}/model")
    mlflow_run_id = Column(String, nullable=False, unique=True, index=True)

    # MLflow Model Registry integration
    mlflow_registry_name = Column(String, nullable=True)
    mlflow_version = Column(String, nullable=True)

    # Metadatos como: {"framework": "pytorch", "architecture": "unet", "num_classes": 2}
    metrics_metadata = Column(JSON, nullable=True)

    # Object key del pipeline de preprocesamiento en el StorageService
    preprocessing_pipeline_path = Column(String, nullable=True)

    is_active = Column(Boolean, default=True)
    is_public = Column(Boolean, default=False)

    # Versioning (semantic versioning)
    version = Column(String, nullable=False, default="1.0.0")

    # Stage in MLflow Registry (Staging, Production, Archived)
    stage = Column(String, nullable=False, default=ModelStage.STAGING.value)

    # Promotion tracking
    promoted_at = Column(DateTime, nullable=True)
    promoted_by = Column(String, nullable=True)

    # Soporte para TorchScript
    is_torchscript = Column(Boolean, default=False)
    # Object key del modelo TorchScript exportado (.pt) en el StorageService
    torchscript_path = Column(String, nullable=True)

    # Aislamiento
    tenant_id = Column(String, ForeignKey("tenant.id"), nullable=False, index=True)
    tenant = relationship("Tenant", back_populates="models")

    # Trazabilidad DVC
    dataset_dvc_hash = Column(String, nullable=True)
    dataset_dvc_registry_name = Column(String, nullable=True)
