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
    """
    id = Column(String, primary_key=True, default=generate_uuid, index=True)
    name = Column(String, nullable=False)
    description = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # ID del run en MLFlow para cargarlo vía mlflow.pytorch.load_model(f"runs:/{run_id}/model")
    mlflow_run_id = Column(String, nullable=False, unique=True, index=True)
    
    # Metadatos como: {"framework": "pytorch", "architecture": "unet", "num_classes": 2}
    metrics_metadata = Column(JSON, nullable=True)
    
    preprocessing_pipeline_path = Column(String, nullable=True) # Ruta al pipeline de preprocesamiento asociado
    
    is_active = Column(Boolean, default=True)
    is_public = Column(Boolean, default=False) # Si es público todos pueden usarlo, si no, solo el tenant.
    
    # Soporte para TorchScript
    is_torchscript = Column(Boolean, default=False)
    torchscript_path = Column(String, nullable=True) # file path si se subió un .pt
    
    # Aislamiento
    tenant_id = Column(String, ForeignKey("tenant.id"), nullable=False, index=True)
    tenant = relationship("Tenant", back_populates="models")
