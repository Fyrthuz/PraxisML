from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.api.deps import get_current_tenant, require_viewer
from app.database import get_db
from app.models.user import User
from app.models.tenant import Tenant
from app.models.dataset import Dataset
from app.core_ml.tabular_parser import read_tabular, is_tabular
from app.services.data_profiler import profile_dataset
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/{dataset_id}/profile", response_model=dict)
def get_dataset_profile(
    dataset_id: str,
    _user: User = Depends(require_viewer),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """
    Returns the statistical profile of each column in a tabular dataset.
    Includes type, null counts, min/max/mean/std for numeric, and value counts for categorical.
    Requiere rol **viewer** o superior.
    """
    dataset = db.query(Dataset).filter(
        Dataset.id == dataset_id, Dataset.tenant_id == tenant.id
    ).first()

    if not dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found or access denied.")

    if not dataset.file_type or not is_tabular(dataset.file_type):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Profiling is only available for tabular datasets."
        )

    try:
        df = read_tabular(dataset.file_path, dataset.file_type)
        profile = profile_dataset(df)
        return {
            "dataset_id": dataset.id,
            "dataset_name": dataset.name,
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "profile": profile
        }
    except Exception as e:
        logger.exception(f"Error profiling dataset {dataset.id}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to profile dataset: {str(e)}")
