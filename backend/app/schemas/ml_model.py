from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict


class ModelStage(str, Enum):
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"


class MLModelBase(BaseModel):
    name: str
    description: Optional[str] = None
    mlflow_run_id: Optional[str] = None
    metrics_metadata: Optional[Dict[str, Any]] = None
    is_public: bool = False
    is_torchscript: bool = False
    torchscript_path: Optional[str] = None
    version: str = "1.0.0"
    stage: ModelStage = ModelStage.STAGING


class MLModelCreate(MLModelBase):
    pass


class MLModelResponse(MLModelBase):
    id: str
    created_at: Optional[datetime] = None
    tenant_id: str
    is_active: bool
    promoted_at: Optional[datetime] = None
    promoted_by: Optional[str] = None
    mlflow_registry_name: Optional[str] = None
    mlflow_version: Optional[str] = None

    from pydantic import field_validator
    @field_validator("mlflow_version", mode="before")
    def coerce_version_to_str(cls, v):
        return str(v) if v is not None else None

    model_config = ConfigDict(from_attributes=True)
