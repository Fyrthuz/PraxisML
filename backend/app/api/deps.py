"""
Dependencias de FastAPI para autenticación, autorización RBAC y quota limiting.

Uso:
    # Solo autenticación
    user = Depends(get_current_user)

    # Con rol mínimo requerido
    user = Depends(require_role("editor"))
    user = Depends(require_admin)

    # Tenant completo
    tenant = Depends(get_current_tenant)

    # Quota checks (usar en endpoints de creación)
    _ = Depends(check_dataset_quota)
    _ = Depends(check_model_quota)
    _ = Depends(check_prediction_quota)
    _ = Depends(check_training_quota)
"""

from datetime import datetime, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core import security
from app.core.exceptions import QuotaExceededError
from app.database import get_db
from app.models.tenant import Tenant
from app.models.user import User, UserRole

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


def get_current_user(
    token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)
) -> User:
    """Valida el JWT y retorna el User activo de la BD."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    # Intenta decodificar de Custom Secret Key o desde JWKS Externo (Clerk/Auth0)
    payload = security.decode_token(token)
    if payload is None:
        raise credentials_exception

    user_id: str = payload.get("sub")
    if user_id is None:
        raise credentials_exception

    user = db.query(User).filter(User.id == user_id).first()

    # Si tenemos Clerk y el usuario no existe, podríamos crearlo On The Fly aquí
    # asumiendo que el ID de Clerk es el `sub` y extrayendo un tenant_id base.
    # Por ahora simplemente fallaremos si el User no está en DB.

    if user is None:
        raise credentials_exception

    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")

    return user


def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    return current_user


def get_current_tenant(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Tenant:
    """Extrae el tenant completo asociado al usuario."""
    tenant = db.query(Tenant).filter(Tenant.id == current_user.tenant_id).first()
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found for user")
    return tenant


# ── RBAC helpers ──────────────────────────────────────────────────────────────

_ROLE_HIERARCHY = {
    UserRole.ADMIN: 3,
    UserRole.EDITOR: 2,
    UserRole.VIEWER: 1,
}


def require_role(minimum_role: str):
    """
    Dependency factory que verifica que el usuario tenga al menos el rol indicado.

    Jerarquía: admin > editor > viewer

    Uso:
        @router.post("/train")
        def train(..., user: User = Depends(require_role("editor"))):
            ...
    """

    def checker(current_user: User = Depends(get_current_user)) -> User:
        user_level = _ROLE_HIERARCHY.get(current_user.role, 0)
        required_level = _ROLE_HIERARCHY.get(minimum_role, 99)
        if user_level < required_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Se requiere rol '{minimum_role}' o superior. Tu rol actual es '{current_user.role}'.",
            )
        return current_user

    return checker


# Shorthands más expresivos
require_admin = require_role(UserRole.ADMIN)
require_editor = require_role(UserRole.EDITOR)
require_viewer = require_role(UserRole.VIEWER)


# ── Quota Limiting ────────────────────────────────────────────────────────────


def check_dataset_quota(
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
) -> Tenant:
    """Verifica que el tenant no haya excedido su cuota de datasets."""
    if tenant.max_datasets is not None:
        from app.models.dataset import Dataset

        current_count = (
            db.query(func.count(Dataset.id))
            .filter(Dataset.tenant_id == tenant.id)
            .scalar()
        )
        if current_count >= tenant.max_datasets:
            raise QuotaExceededError(
                resource="datasets",
                current=current_count,
                limit=tenant.max_datasets,
            )
    return tenant


def check_model_quota(
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
) -> Tenant:
    """Verifica que el tenant no haya excedido su cuota de modelos."""
    if tenant.max_models is not None:
        from app.models.ml_model import MLModel

        current_count = (
            db.query(func.count(MLModel.id))
            .filter(MLModel.tenant_id == tenant.id)
            .scalar()
        )
        if current_count >= tenant.max_models:
            raise QuotaExceededError(
                resource="modelos",
                current=current_count,
                limit=tenant.max_models,
            )
    return tenant


def check_prediction_quota(
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
) -> Tenant:
    """Verifica que el tenant no haya excedido su cuota diaria de predicciones."""
    if tenant.max_predictions_per_day is not None:
        from app.models.prediction import Prediction

        today_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        today_count = (
            db.query(func.count(Prediction.id))
            .filter(
                Prediction.tenant_id == tenant.id,
                Prediction.created_at >= today_start,
            )
            .scalar()
        )
        if today_count >= tenant.max_predictions_per_day:
            raise QuotaExceededError(
                resource="predicciones diarias",
                current=today_count,
                limit=tenant.max_predictions_per_day,
            )
    return tenant


def check_training_quota(
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
) -> Tenant:
    """Verifica que el tenant no haya excedido su cuota diaria de entrenamientos."""
    if tenant.max_training_jobs_per_day is not None:
        from app.models.ml_model import MLModel

        today_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        today_count = (
            db.query(func.count(MLModel.id))
            .filter(
                MLModel.tenant_id == tenant.id,
                MLModel.created_at >= today_start,
            )
            .scalar()
        )
        if today_count >= tenant.max_training_jobs_per_day:
            raise QuotaExceededError(
                resource="entrenamientos diarios",
                current=today_count,
                limit=tenant.max_training_jobs_per_day,
            )
    return tenant
