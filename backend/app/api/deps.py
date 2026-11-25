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

import hashlib
from datetime import datetime, timezone
from typing import Optional

from fastapi import Depends, Header, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core import security
from app.core.exceptions import QuotaExceededError
from app.core.rls import set_tenant_rls_context
from app.database import get_db
from app.models.api_key import APIKey
from app.models.tenant import Tenant
from app.models.user import User, UserRole
from app.services.storage_service import StorageService, get_storage

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)


def _authenticate_api_key(raw_key: str, db: Session, request: Optional[Request] = None) -> Optional[User]:
    """Valida una API key (`px_...`) mediante SHA-256 y devuelve el usuario propietario."""
    key_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
    api_key = (
        db.query(APIKey)
        .filter(APIKey.key_hash_sha256 == key_hash, APIKey.revoked_at.is_(None))
        .first()
    )
    if api_key is None:
        return None

    now = datetime.now(timezone.utc)
    if api_key.expires_at is not None:
        exp = api_key.expires_at
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        if exp < now:
            return None

    user = db.query(User).filter(User.id == api_key.user_id).first()
    if user is None or not user.is_active:
        return None

    api_key.last_used_at = now
    try:
        db.commit()
    except Exception:
        db.rollback()

    if request is not None:
        request.state.auth_type = "api_key"
        request.state.api_key_id = api_key.id
        request.state.api_key_scopes = api_key.scopes or ["*"]
        request.state.tenant_id = user.tenant_id

    return user


def get_current_user(
    request: Request,
    token: Optional[str] = Depends(oauth2_scheme),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    db: Session = Depends(get_db),
) -> User:
    """Valida JWT Bearer o API Key (`X-API-Key` / `Bearer px_...`) y retorna el User activo."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    candidate_api_key = x_api_key or (token if token and token.startswith("px_") else None)
    if candidate_api_key:
        user = _authenticate_api_key(candidate_api_key, db, request=request)
        if user is None:
            raise credentials_exception
        return user

    if not token:
        raise credentials_exception

    payload = security.decode_token(token)
    if payload is None:
        raise credentials_exception

    user_id: str = payload.get("sub")
    if user_id is None:
        raise credentials_exception

    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise credentials_exception

    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")

    request.state.auth_type = "jwt"
    request.state.tenant_id = user.tenant_id
    return user


def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    return current_user


def get_current_tenant(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Tenant:
    """Extrae el tenant asociado al usuario y configura el contexto RLS en PostgreSQL."""
    tenant = db.query(Tenant).filter(Tenant.id == current_user.tenant_id).first()
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found for user")
    set_tenant_rls_context(db, tenant.id)
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


def _update_quota_gauge(tenant_id: str, resource: str, current: int, limit: int) -> None:
    from app.observability.metrics import quota_usage_pct

    pct = (current / limit) * 100 if limit > 0 else 0
    quota_usage_pct.labels(tenant_id=tenant_id, resource=resource).set(pct)


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
        _update_quota_gauge(tenant.id, "datasets", current_count, tenant.max_datasets)
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
        _update_quota_gauge(tenant.id, "modelos", current_count, tenant.max_models)
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
        _update_quota_gauge(tenant.id, "predicciones", today_count, tenant.max_predictions_per_day)
        if today_count >= tenant.max_predictions_per_day:
            raise QuotaExceededError(
                resource="predicciones diarias",
                current=today_count,
                limit=tenant.max_predictions_per_day,
            )
    return tenant


def get_storage_service() -> StorageService:
    """FastAPI dependency que retorna la instancia de StorageService."""
    return get_storage()


# ── Observability Dependencies ────────────────────────────────────────────────


def get_usage_collector_dep():
    """FastAPI dependency que retorna el singleton UsageCollector."""
    from app.observability.usage_collector import get_usage_collector

    return get_usage_collector()


def get_cost_calculator_dep():
    """FastAPI dependency que retorna el singleton CostCalculator."""
    from app.observability.cost_calculator import get_cost_calculator

    return get_cost_calculator()


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
        _update_quota_gauge(tenant.id, "entrenamientos", today_count, tenant.max_training_jobs_per_day)
        if today_count >= tenant.max_training_jobs_per_day:
            raise QuotaExceededError(
                resource="entrenamientos diarios",
                current=today_count,
                limit=tenant.max_training_jobs_per_day,
            )
    return tenant
