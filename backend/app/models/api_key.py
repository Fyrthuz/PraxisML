from datetime import datetime, timezone

from sqlalchemy import JSON, Column, DateTime, ForeignKey, String

from app.models.base import Base
from app.utils.uuid import generate_uuid


class APIKey(Base):
    """Personal Access Token / Service Principal API key con scopes y expiración."""

    __tablename__ = "api_keys"

    id = Column(String, primary_key=True, default=generate_uuid, index=True)
    tenant_id = Column(String, ForeignKey("tenant.id"), nullable=False, index=True)
    workspace_id = Column(String, ForeignKey("workspaces.id"), nullable=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    name = Column(String, nullable=False)
    key_prefix = Column(String(20), nullable=False)
    key_hash_sha256 = Column(String(64), nullable=False, unique=True, index=True)
    scopes = Column(JSON, nullable=False, default=lambda: ["*"])
    expires_at = Column(DateTime, nullable=True)
    last_used_at = Column(DateTime, nullable=True)
    revoked_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
