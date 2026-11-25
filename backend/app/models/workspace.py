from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, String, UniqueConstraint

from app.models.base import Base
from app.utils.uuid import generate_uuid


class Workspace(Base):
    """Espacio de trabajo aislado dentro de un tenant (dev / staging / prod)."""

    __tablename__ = "workspaces"
    __table_args__ = (
        UniqueConstraint("tenant_id", "slug", name="uq_workspaces_tenant_slug"),
    )

    id = Column(String, primary_key=True, default=generate_uuid, index=True)
    tenant_id = Column(String, ForeignKey("tenant.id"), nullable=False, index=True)
    name = Column(String, nullable=False)
    slug = Column(String, nullable=False, index=True)
    environment = Column(String, nullable=False, default="dev")
    default_compute_profile = Column(String, nullable=False, default="cpu-light")
    budget_limit_usd = Column(Float, nullable=True)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
