from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, String
from sqlalchemy.orm import relationship

from app.models.base import Base
from app.utils.uuid import generate_uuid


class UserRole:
    """Roles disponibles por tenant."""

    ADMIN = "admin"  # Gestión completa: modelos, datasets, usuarios del tenant
    EDITOR = "editor"  # Puede entrenar, hacer inferencia y gestionar datasets
    VIEWER = "viewer"  # Solo lectura: ver predicciones y modelos


VALID_ROLES = {UserRole.ADMIN, UserRole.EDITOR, UserRole.VIEWER}


class User(Base):
    __tablename__ = "users"  # Explicit: "user" is a reserved keyword in PostgreSQL

    id = Column(String, primary_key=True, default=generate_uuid, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=True)  # Será nulo si usamos Clerk/Auth0
    full_name = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)

    # ── RBAC: rol del usuario dentro de su tenant ────────────────────────────
    # Valores posibles: admin | editor | viewer  (ver UserRole)
    role = Column(String, nullable=False, default=UserRole.VIEWER)

    # Campo CRÍTICO: El usuario pertenece a un Tenant siempre
    tenant_id = Column(String, ForeignKey("tenant.id"), nullable=False, index=True)

    tenant = relationship("Tenant", back_populates="users")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
