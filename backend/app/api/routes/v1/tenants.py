import os
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import (
    get_current_tenant,
    require_admin,
    require_viewer,
)
from app.database import get_db
from app.models.tenant import Tenant
from app.models.user import User
from app.schemas.tenant import TenantCreate, TenantResponse, TenantUpdate

router = APIRouter()


@router.post("/", response_model=TenantResponse, status_code=status.HTTP_201_CREATED)
def create_tenant(
    tenant_in: TenantCreate,
    user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """
    Crea un nuevo tenant. Solo admins pueden crear tenants.
    """
    # 1. Crear registro en BD
    new_tenant = Tenant(name=tenant_in.name)
    db.add(new_tenant)
    db.commit()
    db.refresh(new_tenant)

    # 2. Actualizar el tenant_id del usuario creador (MVP: 1 usuario = 1 tenant)
    user.tenant_id = new_tenant.id
    db.commit()

    # 3. Crear carpeta local para almacenar sus archivos privados
    from app.core.config import settings
    tenant_dir = os.path.join(settings.DATA_DIR, "tenants", new_tenant.id, "datasets")
    os.makedirs(tenant_dir, exist_ok=True)

    return new_tenant


@router.get("/me", response_model=TenantResponse)
def get_my_active_tenant(
    tenant: Tenant = Depends(get_current_tenant),
    _user: User = Depends(require_viewer),
    db: Session = Depends(get_db),
):
    """Retorna el tenant actual del usuario autenticado (basado en token)."""
    return tenant


@router.get("/my_tenants", response_model=List[TenantResponse])
def get_my_tenants(
    user: User = Depends(require_viewer),
    db: Session = Depends(get_db),
):
    """Lista todos los tenants a los que pertenece el usuario (MVP: solo 1 activo)."""
    tenant = db.query(Tenant).filter(Tenant.id == user.tenant_id).first()
    return [tenant] if tenant else []


@router.patch("/{tenant_id}", response_model=TenantResponse)
def update_tenant_quotas(
    tenant_id: str,
    update_in: TenantUpdate,
    user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """
    Actualiza las cuotas de un tenant. Solo admins del tenant pueden hacerlo.
    """
    tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
    if not tenant:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tenant no encontrado")

    # Verificar que el admin pertenece a este tenant
    if user.tenant_id != tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="No tienes permisos para modificar este tenant.",
        )

    update_data = update_in.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(tenant, field, value)

    db.commit()
    db.refresh(tenant)
    return tenant


@router.get("/{tenant_id}", response_model=TenantResponse)
def get_tenant(
    tenant_id: str,
    user: User = Depends(require_viewer),
    db: Session = Depends(get_db),
):
    tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
    if not tenant:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tenant no encontrado")
    return tenant
