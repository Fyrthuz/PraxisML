from sqlalchemy import Column, String, DateTime, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from app.models.base import Base
import uuid

def generate_uuid():
    return str(uuid.uuid4())

class Tenant(Base):
    id = Column(String, primary_key=True, default=generate_uuid, index=True)
    name = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Un tenant tiene muchos usuarios y muchos datasets (relaciones)
    users = relationship("User", back_populates="tenant")
    datasets = relationship("Dataset", back_populates="tenant", cascade="all, delete-orphan")
    models = relationship("MLModel", back_populates="tenant", cascade="all, delete-orphan")
