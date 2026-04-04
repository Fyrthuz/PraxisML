from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class PredictionResponse(BaseModel):
    id: str
    task_id: Optional[str] = None
    status: str
    method: str
    result_path: Optional[str] = None
    uncertainty_path: Optional[str] = None
    mlflow_inference_run_id: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    dataset_id: Optional[str] = None
    model_id: str
    tenant_id: str

    model_config = ConfigDict(from_attributes=True)
