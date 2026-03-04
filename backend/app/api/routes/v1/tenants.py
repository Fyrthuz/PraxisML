from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.tenant import Tenant
from app.models.user import User
from app.schemas.tenant import TenantCreate, TenantResponse
from typing import List
from app.api.deps import get_current_active_user, get_current_tenant
import os

router = APIRouter()

@router.post("/", response_model=TenantResponse, status_code=status.HTTP_201_CREATED)
def create_tenant(
    tenant_in: TenantCreate, 
    user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
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
def get_my_active_tenant(tenant: Tenant = Depends(get_current_tenant), db: Session = Depends(get_db)):
    """Retorna el tenant actual del usuario autenticado (basado en token)."""
    return tenant

@router.get("/my_tenants", response_model=List[TenantResponse])
def get_my_tenants(user: User = Depends(get_current_active_user), db: Session = Depends(get_db)):
    """Lista todos los tenants a los que pertenece el usuario (MVP: solo 1 activo)."""
    tenant = db.query(Tenant).filter(Tenant.id == user.tenant_id).first()
    return [tenant] if tenant else []

@router.get("/{tenant_id}", response_model=TenantResponse)
def get_tenant(
    tenant_id: str, 
    user: User = Depends(get_current_active_user), 
    db: Session = Depends(get_db)
):
    tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
    if not tenant:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tenant no encontrado")
    return tenant
