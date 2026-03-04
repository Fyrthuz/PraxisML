from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class TenantBase(BaseModel):
    name: str

class TenantCreate(TenantBase):
    pass

class TenantResponse(TenantBase):
    id: str
    created_at: datetime
    is_active: bool

    model_config = {"from_attributes": True}
