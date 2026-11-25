from datetime import datetime, timezone

from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint

from app.models.base import Base
from app.utils.uuid import generate_uuid


class UserPipeline(Base):
    """
    Pipeline reutilizable definido por el usuario y vinculado a un RuntimeEnvironment.
    Soporta pasos declarativos (sklearn/plugins), código Python personalizado `transform(df)`
    o combinación híbrida aplicable a cualquier dataset.
    """

    __tablename__ = "user_pipelines"
    __table_args__ = (
        UniqueConstraint("tenant_id", "slug", name="uq_user_pipelines_tenant_slug"),
    )

    id = Column(String, primary_key=True, default=generate_uuid, index=True)
    tenant_id = Column(String, ForeignKey("tenant.id"), nullable=False, index=True)
    workspace_id = Column(String, ForeignKey("workspaces.id"), nullable=True, index=True)
    environment_id = Column(String, ForeignKey("runtime_environments.id"), nullable=True, index=True)

    name = Column(String, nullable=False)
    slug = Column(String, nullable=False, index=True)
    description = Column(String, nullable=True)

    engine = Column(String, nullable=False, default="declarative")  # declarative | python | hybrid
    steps = Column(JSON, nullable=False, default=list)
    custom_code = Column(Text, nullable=True)
    column_renames = Column(JSON, nullable=False, default=dict)
    target_column = Column(String, nullable=True)
    version = Column(Integer, nullable=False, default=1)

    created_by = Column(String, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
