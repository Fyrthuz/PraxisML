"""
Modelo ORM para conversaciones del asistente LLM (RAG).
"""

from datetime import datetime, timezone

from sqlalchemy import JSON, Column, DateTime, String

from app.models.base import Base
from app.utils.uuid import generate_uuid


class AssistantConversation(Base):
    __tablename__ = "assistant_conversations"

    id = Column(String, primary_key=True, default=generate_uuid, index=True)
    tenant_id = Column(String, nullable=False, index=True)
    user_id = Column(String, nullable=True)
    # [{role: "user"|"assistant", content: str, ts: ISO8601}]
    messages = Column(JSON, nullable=False, default=list)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
