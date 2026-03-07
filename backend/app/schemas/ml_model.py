from typing import Optional, Any, Dict
from datetime import datetime
from pydantic import BaseModel, ConfigDict

class MLModelBase(BaseModel):
    name: str
    description: Optional[str] = None
    mlflow_run_id: Optional[str] = None # Optional now, since it could just be a .pt file
    metrics_metadata: Optional[Dict[str, Any]] = None
    is_public: bool = False
    is_torchscript: bool = False
    torchscript_path: Optional[str] = None

class MLModelCreate(MLModelBase):
    pass

class MLModelResponse(MLModelBase):
    id: str
    created_at: Optional[datetime] = None
    tenant_id: str
    is_active: bool

    model_config = ConfigDict(from_attributes=True)
