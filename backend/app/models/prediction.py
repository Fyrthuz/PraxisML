from sqlalchemy import Column, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from app.models.base import Base
import uuid


def generate_uuid():
    return str(uuid.uuid4())


class Prediction(Base):
    """
    Almacena los resultados de cada inferencia lanzada vía Celery.
    El ciclo de vida es: PENDING → RUNNING → COMPLETED | FAILED.

    Las columnas *_path guardan object keys del StorageService
    (e.g. "{tenant_id}/predictions/{id}.npy"), NO rutas de disco.
    Usa get_storage().download(key) para obtener el contenido.
    """

    id = Column(String, primary_key=True, default=generate_uuid, index=True)

    # Celery task ID para hacer polling del estado
    task_id = Column(String, nullable=True, unique=True, index=True)

    # Estado de la inferencia
    status = Column(
        String, nullable=False, default="PENDING"
    )  # PENDING/RUNNING/COMPLETED/FAILED

    # Método de incertidumbre utilizado
    method = Column(String, nullable=False, default="mc_dropout")

    # Object keys en el StorageService (formato: "{tenant_id}/predictions/{prediction_id}/{archivo}")
    result_path = Column(String, nullable=True)  # prediction probabilities .npy
    uncertainty_path = Column(String, nullable=True)  # uncertainty map .npy
    input_image_path = Column(String, nullable=True)  # imagen de entrada original

    # MLFlow inference tracking run ID
    mlflow_inference_run_id = Column(String, nullable=True, index=True)

    # Mensaje de error si algo falla
    error_message = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)

    # Relaciones FK
    dataset_id = Column(String, ForeignKey("dataset.id"), nullable=True, index=True)
    model_id = Column(String, ForeignKey("ml_model.id"), nullable=False, index=True)
    tenant_id = Column(String, ForeignKey("tenant.id"), nullable=False, index=True)

    dataset = relationship("Dataset")
    ml_model = relationship("MLModel")
    tenant = relationship("Tenant")
