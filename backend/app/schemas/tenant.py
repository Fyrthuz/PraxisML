from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class TenantBase(BaseModel):
    name: str

class TenantCreate(TenantBase):
    pass

class TenantUpdate(BaseModel):
    """Schema para actualizar cuotas de un tenant (solo admin)."""
    name: Optional[str] = None
    max_datasets: Optional[int] = None
    max_models: Optional[int] = None
    max_predictions_per_day: Optional[int] = None
    max_training_jobs_per_day: Optional[int] = None

class TenantResponse(TenantBase):
    id: str
    created_at: datetime
    is_active: bool
    max_datasets: Optional[int] = None
    max_models: Optional[int] = None
    max_predictions_per_day: Optional[int] = None
    max_training_jobs_per_day: Optional[int] = None

    model_config = {"from_attributes": True}
