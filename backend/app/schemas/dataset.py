from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, ConfigDict


class DatasetBase(BaseModel):
    name: str
    description: Optional[str] = None


class DatasetCreate(DatasetBase):
    pass


class DatasetResponse(DatasetBase):
    id: str
    file_path: str
    config_path: Optional[str] = None
    file_size_bytes: Optional[int] = None
    created_at: datetime
    tenant_id: str

    # ── Nuevos campos Fase 1 ──────────────────────────────────────────────────
    file_type: Optional[str] = None
    num_rows: Optional[int] = None
    num_columns: Optional[int] = None
    column_names: Optional[List[str]] = None
    version: int = 1
    mlflow_artifact_uri: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class DatasetPreviewResponse(BaseModel):
    """Respuesta para el endpoint de preview de datasets tabulares."""
    dataset_id: str
    file_type: str
    num_rows: int
    num_columns: int
    column_names: List[str]
    column_dtypes: dict[str, str]
    preview_rows: List[dict]  # First N rows as list of dicts
