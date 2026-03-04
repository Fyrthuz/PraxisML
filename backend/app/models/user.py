from sqlalchemy import Column, String, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from app.models.base import Base
import uuid

def generate_uuid():
    return str(uuid.uuid4())

class User(Base):
    __tablename__ = "users"  # Explicit: "user" is a reserved keyword in PostgreSQL

    id = Column(String, primary_key=True, default=generate_uuid, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=True) # Será nulo si usamos Clerk/Auth0
    full_name = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    
    # Campo CRÍTICO: El usuario pertenece a un Tenant siempre
    tenant_id = Column(String, ForeignKey("tenant.id"), nullable=False, index=True)
    
    tenant = relationship("Tenant", back_populates="users")
