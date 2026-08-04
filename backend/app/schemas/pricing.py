"""
Schemas Pydantic para la gestión dinámica de precios (resource_pricing).
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ResourcePricingCreate(BaseModel):
    resource_category: str = Field(
        ..., description="compute_gpu | compute_cpu | storage | llm_tokens"
    )
    resource_id: str = Field(..., description="A100 | gpt-4 | standard | ...")
    pricing_unit: str = Field(..., description="per_hour | per_1k_input_tokens | per_gb_month")
    price_per_unit: float = Field(..., gt=0)
    currency: str = "USD"
    metadata: Optional[dict] = None


class ResourcePricingUpdate(BaseModel):
    price_per_unit: float = Field(..., gt=0)
    metadata: Optional[dict] = None


class ResourcePricingResponse(BaseModel):
    id: str
    resource_category: str
    resource_id: str
    pricing_unit: str
    price_per_unit: float
    currency: str
    metadata: Optional[dict] = Field(None, validation_alias="extra_metadata")
    effective_from: datetime
    effective_to: Optional[datetime] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ResourcePricingHistoryResponse(BaseModel):
    history: list[ResourcePricingResponse]
