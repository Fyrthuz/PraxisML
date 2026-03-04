from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from app.database import get_db
from app.core import security
from app.models.user import User
from app.models.tenant import Tenant

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
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

def get_current_active_user(current_user: User = Depends(get_current_user)):
    return current_user

def get_current_tenant(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Extrae el tenant completo asociado al usuario"""
    tenant = db.query(Tenant).filter(Tenant.id == current_user.tenant_id).first()
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found for user")
    return tenant
