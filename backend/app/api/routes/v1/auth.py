from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pydantic import BaseModel, ConfigDict, EmailStr
from app.database import get_db
from app.core import security
from app.models.user import User, UserRole
from app.models.tenant import Tenant
from app.api.deps import get_current_active_user

router = APIRouter()

class Token(BaseModel):
    access_token: str
    token_type: str

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: str | None = None
    # Permitir crear un nuevo tenant en el registro
    tenant_name: str | None = None

class UserResponse(BaseModel):
    id: str
    email: EmailStr
    full_name: str | None
    tenant_id: str
    role: str

    model_config = ConfigDict(from_attributes=True)

@router.post("/register", response_model=UserResponse)
def register(user_in: UserCreate, db: Session = Depends(get_db)):
    """
    Registra un usuario nuevo.
    Si se proporciona `tenant_name`, crea un nuevo tenant y lo asocia.
    El primer usuario de un tenant obtiene rol 'admin' automáticamente.
    """
    user = db.query(User).filter(User.email == user_in.email).first()
    if user:
        raise HTTPException(
            status_code=400,
            detail="The user with this email already exists in the system.",
        )

    # Crear tenant si no existe nada (MVP Mode)
    tenant_title = user_in.tenant_name if user_in.tenant_name else f"Tenant of {user_in.email}"
    new_tenant = Tenant(name=tenant_title)
    db.add(new_tenant)
    db.commit()
    db.refresh(new_tenant)

    # El primer usuario de un tenant es admin
    user = User(
        email=user_in.email,
        hashed_password=security.get_password_hash(user_in.password),
        full_name=user_in.full_name,
        tenant_id=new_tenant.id,
        role=UserRole.ADMIN,  # Primer usuario = admin del tenant
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

@router.post("/login", response_model=Token)
def login_access_token(
    db: Session = Depends(get_db), form_data: OAuth2PasswordRequestForm = Depends()
):
    """
    OAuth2 compatible token login, get an access token for future requests
    """
    # Try looking by email since we use email as the username
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect email or password")

    # Para passwords de Auth0/Clerk, su hashed_password en DB sería None y no pasarían por aquí,
    # el cliente pediría el Token de Auth0 y usaría dicho token. Esto es solo para Custom Auth.
    if not user.hashed_password or not security.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect email or password")

    # Generar Token JWT con id de usuario y tenant_id
    access_token = security.create_access_token(
        subject=user.id,
        extra_data={"tenant_id": user.tenant_id}
    )
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/me", response_model=UserResponse)
def get_current_user_profile(current_user: User = Depends(get_current_active_user)):
    """Retorna información del usuario logueado en base al JWT validado."""
    return current_user
