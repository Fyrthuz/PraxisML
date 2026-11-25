from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, Column, DateTime, ForeignKey, String, UniqueConstraint

from app.models.base import Base
from app.utils.uuid import generate_uuid


class RuntimeEnvironment(Base):
    """
    Entorno de ejecución configurable estilo Databricks (versión de Python,
    imagen base, paquetes PyPI/uv, variables de entorno y perfil de compute).
    """

    __tablename__ = "runtime_environments"
    __table_args__ = (
        UniqueConstraint("tenant_id", "slug", name="uq_runtime_env_tenant_slug"),
    )

    id = Column(String, primary_key=True, default=generate_uuid, index=True)
    tenant_id = Column(String, ForeignKey("tenant.id"), nullable=False, index=True)
    workspace_id = Column(String, ForeignKey("workspaces.id"), nullable=True, index=True)
    name = Column(String, nullable=False)
    slug = Column(String, nullable=False, index=True)
    description = Column(String, nullable=True)

    python_version = Column(String, nullable=False, default="3.10")
    base_image = Column(String, nullable=False, default="python:3.10-slim")
    packages = Column(JSON, nullable=False, default=list)
    env_vars = Column(JSON, nullable=False, default=dict)
    compute_profile = Column(String, nullable=False, default="cpu-heavy")
    runner_type = Column(String, nullable=False, default="uv_isolated")
    is_default = Column(Boolean, nullable=False, default=False)

    created_by = Column(String, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
