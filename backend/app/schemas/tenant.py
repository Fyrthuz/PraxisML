from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class TenantBase(BaseModel):
    name: str

class TenantCreate(TenantBase):
    pass

class TenantUpdate(BaseModel):
    """Schema para actualizar cuotas y budget de un tenant (solo admin)."""
    name: Optional[str] = None
    max_datasets: Optional[int] = None
    max_models: Optional[int] = None
    max_predictions_per_day: Optional[int] = None
    max_training_jobs_per_day: Optional[int] = None
    monthly_budget_usd: Optional[float] = None
    budget_alert_pct: Optional[float] = None
    billing_cycle: Optional[str] = None
    stripe_customer_id: Optional[str] = None

class TenantResponse(TenantBase):
    id: str
    created_at: datetime
    is_active: bool
    max_datasets: Optional[int] = None
    max_models: Optional[int] = None
    max_predictions_per_day: Optional[int] = None
    max_training_jobs_per_day: Optional[int] = None
    monthly_budget_usd: Optional[float] = None
    budget_alert_pct: Optional[float] = None
    billing_cycle: Optional[str] = None
    stripe_customer_id: Optional[str] = None

    model_config = {"from_attributes": True}
