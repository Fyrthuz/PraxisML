from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session

from app.api.deps import get_current_tenant, require_admin
from app.core import security
from app.database import get_db
from app.models.tenant import Tenant
from app.models.user import VALID_ROLES, User

router = APIRouter()


class UserResponse(BaseModel):
    id: str
    email: str
    full_name: str | None
    role: str
    is_active: bool
    tenant_id: str
    created_at: str

    class Config:
        from_attributes = True


class CreateUserRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    full_name: str | None = None
    role: str = Field(default="viewer", pattern="^(admin|editor|viewer)$")


class UpdateUserRoleRequest(BaseModel):
    role: str


@router.get("/", response_model=List[UserResponse])
def list_users(
    db: Session = Depends(get_db),
    _user: User = Depends(require_admin),
    _tenant: Tenant = Depends(get_current_tenant),
):
    """Lista todos los usuarios del tenant. Solo admins pueden ver la lista."""
    users = db.query(User).filter(User.tenant_id == _tenant.id).all()
    return [
        UserResponse(
            id=u.id,
            email=u.email,
            full_name=u.full_name,
            role=u.role,
            is_active=u.is_active,
            tenant_id=u.tenant_id,
            created_at=u.created_at.isoformat() if u.created_at else "",
        )
        for u in users
    ]


@router.get("/{user_id}", response_model=UserResponse)
def get_user(
    user_id: str,
    db: Session = Depends(get_db),
    _user: User = Depends(require_admin),
    _tenant: Tenant = Depends(get_current_tenant),
):
    """Obtiene un usuario específico del tenant. Solo admins."""
    target_user = (
        db.query(User).filter(User.id == user_id, User.tenant_id == _tenant.id).first()
    )
    if not target_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Usuario {user_id} no encontrado en este tenant",
        )
    return UserResponse(
        id=target_user.id,
        email=target_user.email,
        full_name=target_user.full_name,
        role=target_user.role,
        is_active=target_user.is_active,
        tenant_id=target_user.tenant_id,
        created_at=target_user.created_at.isoformat() if target_user.created_at else "",
    )


@router.patch("/{user_id}/role", response_model=UserResponse)
def update_user_role(
    user_id: str,
    request: UpdateUserRoleRequest,
    db: Session = Depends(get_db),
    _user: User = Depends(require_admin),
    _tenant: Tenant = Depends(get_current_tenant),
):
    """Cambia el rol de un usuario. Solo admins pueden cambiar roles."""
    if request.role not in VALID_ROLES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Rol inválido. Roles válidos: {', '.join(VALID_ROLES)}",
        )

    target_user = (
        db.query(User).filter(User.id == user_id, User.tenant_id == _tenant.id).first()
    )
    if not target_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Usuario {user_id} no encontrado en este tenant",
        )

    if target_user.id == _user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No puedes cambiar tu propio rol",
        )

    target_user.role = request.role
    db.commit()
    db.refresh(target_user)

    return UserResponse(
        id=target_user.id,
        email=target_user.email,
        full_name=target_user.full_name,
        role=target_user.role,
        is_active=target_user.is_active,
        tenant_id=target_user.tenant_id,
        created_at=target_user.created_at.isoformat() if target_user.created_at else "",
    )


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def deactivate_user(
    user_id: str,
    db: Session = Depends(get_db),
    _user: User = Depends(require_admin),
    _tenant: Tenant = Depends(get_current_tenant),
):
    """Desactiva un usuario (soft delete). Solo admins."""
    target_user = (
        db.query(User).filter(User.id == user_id, User.tenant_id == _tenant.id).first()
    )
    if not target_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Usuario {user_id} no encontrado en este tenant",
        )

    if target_user.id == _user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No puedes desactivarte a ti mismo",
        )

    target_user.is_active = False
    db.commit()


@router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def create_user(
    request: CreateUserRequest,
    db: Session = Depends(get_db),
    _user: User = Depends(require_admin),
    _tenant: Tenant = Depends(get_current_tenant),
):
    """Crea un nuevo usuario en el tenant. Solo admins pueden crear usuarios."""
    existing = db.query(User).filter(User.email == request.email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Ya existe un usuario con este email",
        )

    if request.role not in VALID_ROLES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Rol inválido. Roles válidos: {', '.join(VALID_ROLES)}",
        )

    new_user = User(
        email=request.email,
        hashed_password=security.get_password_hash(request.password),
        full_name=request.full_name,
        tenant_id=_tenant.id,
        role=request.role,
        is_active=True,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return UserResponse(
        id=new_user.id,
        email=new_user.email,
        full_name=new_user.full_name,
        role=new_user.role,
        is_active=new_user.is_active,
        tenant_id=new_user.tenant_id,
        created_at=new_user.created_at.isoformat() if new_user.created_at else "",
    )
