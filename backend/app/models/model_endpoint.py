from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String

from app.models.base import Base
from app.utils.uuid import generate_uuid


class ModelEndpoint(Base):
    """Endpoint HTTP dedicado bajo demanda para servir un modelo registrado."""

    __tablename__ = "model_endpoints"

    id = Column(String, primary_key=True, default=generate_uuid, index=True)
    tenant_id = Column(String, ForeignKey("tenant.id"), nullable=False, index=True)
    workspace_id = Column(String, ForeignKey("workspaces.id"), nullable=True, index=True)
    name = Column(String, nullable=False)
    slug = Column(String, nullable=False, unique=True, index=True)
    description = Column(String, nullable=True)

    model_id = Column(String, ForeignKey("ml_model.id"), nullable=False, index=True)
    secondary_model_id = Column(String, ForeignKey("ml_model.id"), nullable=True)
    traffic_split_pct = Column(Integer, nullable=False, default=0)

    preprocessing_pipeline_path = Column(String, nullable=True)
    status = Column(String, nullable=False, default="active")
    auth_mode = Column(String, nullable=False, default="api_key")

    request_count = Column(Integer, nullable=False, default=0)
    error_count = Column(Integer, nullable=False, default=0)
    avg_latency_ms = Column(Float, nullable=False, default=0.0)
    last_invoked_at = Column(DateTime, nullable=True)

    created_by = Column(String, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
