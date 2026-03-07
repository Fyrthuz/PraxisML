from sqlalchemy import Column, String, DateTime, Boolean, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from app.models.base import Base
import uuid

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

    # Metadatos como: {"framework": "pytorch", "architecture": "unet", "num_classes": 2}
    metrics_metadata = Column(JSON, nullable=True)

    # Object key del pipeline de preprocesamiento en el StorageService
    preprocessing_pipeline_path = Column(String, nullable=True)

    is_active = Column(Boolean, default=True)
    is_public = Column(Boolean, default=False)  # Si es público todos pueden usarlo, si no, solo el tenant.

    # Soporte para TorchScript
    is_torchscript = Column(Boolean, default=False)
    # Object key del modelo TorchScript exportado (.pt) en el StorageService
    torchscript_path = Column(String, nullable=True)

    # Aislamiento
    tenant_id = Column(String, ForeignKey("tenant.id"), nullable=False, index=True)
    tenant = relationship("Tenant", back_populates="models")
