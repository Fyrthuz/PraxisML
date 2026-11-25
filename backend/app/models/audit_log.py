from datetime import datetime, timezone

from sqlalchemy import JSON, Column, DateTime, ForeignKey, Index, String

from app.models.base import Base
from app.utils.uuid import generate_uuid


class AuditLog(Base):
    """Registro append-only de acciones de usuarios, API keys, sistema y copilot."""

    __tablename__ = "audit_log"
    __table_args__ = (
        Index("ix_audit_log_tenant_created", "tenant_id", "created_at"),
    )

    id = Column(String, primary_key=True, default=generate_uuid, index=True)
    tenant_id = Column(String, ForeignKey("tenant.id"), nullable=False, index=True)
    workspace_id = Column(String, nullable=True, index=True)
    actor_id = Column(String, nullable=True)
    actor_type = Column(String, nullable=False, default="user")
    action = Column(String, nullable=False, index=True)
    resource_type = Column(String, nullable=False, index=True)
    resource_id = Column(String, nullable=True)
    details = Column(JSON, nullable=True)
    ip_address = Column(String, nullable=True)
    request_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
