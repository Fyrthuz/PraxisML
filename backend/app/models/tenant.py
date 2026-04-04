from sqlalchemy import Column, String, DateTime, Boolean, Integer
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from app.models.base import Base
import uuid


def generate_uuid():
    return str(uuid.uuid4())


class Tenant(Base):
    id = Column(String, primary_key=True, default=generate_uuid, index=True)
    name = Column(String, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    is_active = Column(Boolean, default=True)

    # ── Quota Limiting ────────────────────────────────────────────────────────
    # Límites configurables por tenant. NULL = sin límite.
    max_datasets = Column(Integer, nullable=True, default=50)
    max_models = Column(Integer, nullable=True, default=20)
    max_predictions_per_day = Column(Integer, nullable=True, default=500)
    max_training_jobs_per_day = Column(Integer, nullable=True, default=10)

    # Un tenant tiene muchos usuarios y muchos datasets (relaciones)
    users = relationship("User", back_populates="tenant")
    datasets = relationship(
        "Dataset", back_populates="tenant", cascade="all, delete-orphan"
    )
    models = relationship(
        "MLModel", back_populates="tenant", cascade="all, delete-orphan"
    )
