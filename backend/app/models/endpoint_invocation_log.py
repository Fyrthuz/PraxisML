from datetime import datetime, timezone

from sqlalchemy import JSON, Column, DateTime, Float, ForeignKey, Index, Integer, String

from app.models.base import Base
from app.utils.uuid import generate_uuid


class EndpointInvocationLog(Base):
    """Historial ligero de invocaciones de un ModelEndpoint (base de Inference Tables)."""

    __tablename__ = "endpoint_invocation_logs"
    __table_args__ = (
        Index("ix_endpoint_inv_logs_ep_created", "endpoint_id", "created_at"),
    )

    id = Column(String, primary_key=True, default=generate_uuid, index=True)
    endpoint_id = Column(String, ForeignKey("model_endpoints.id"), nullable=False, index=True)
    tenant_id = Column(String, ForeignKey("tenant.id"), nullable=False, index=True)
    model_id = Column(String, nullable=False)
    variant = Column(String, nullable=False, default="primary")
    runtime = Column(String, nullable=False, default="tabular_generic")
    status = Column(String, nullable=False, default="ok")
    latency_ms = Column(Float, nullable=False, default=0.0)
    input_count = Column(Integer, nullable=False, default=1)
    sample_input = Column(JSON, nullable=True)
    sample_output = Column(JSON, nullable=True)
    error_message = Column(String, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
