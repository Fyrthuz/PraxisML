# Import all models here so SQLAlchemy's Base.metadata.create_all() discovers them.
# This file is the single source of truth for all ORM models.

from app.models.base import Base  # noqa: F401
from app.models.dataset import Dataset  # noqa: F401
from app.models.ml_model import MLModel  # noqa: F401
from app.models.prediction import Prediction  # noqa: F401
from app.models.tenant import Tenant  # noqa: F401
from app.models.user import User  # noqa: F401

__all__ = ["Base", "Tenant", "User", "Dataset", "MLModel", "Prediction"]
