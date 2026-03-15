from datetime import datetime, timedelta
from typing import Any, Union, Dict, Optional
from jose import jwt, JWTError
from passlib.context import CryptContext
import requests
from app.core.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Cache para las JWKS de proveedores externos (simplificado)
_jwks_cache = None
_jwks_last_fetched = None

def get_jwks() -> Optional[Dict]:
    global _jwks_cache, _jwks_last_fetched

    if not settings.EXTERNAL_AUTH_JWKS_URL:
        return None

    # Recargar cada hora la cache de JWKS
    if _jwks_cache is None or _jwks_last_fetched is None or (datetime.utcnow() - _jwks_last_fetched) > timedelta(hours=1):
        try:
            response = requests.get(settings.EXTERNAL_AUTH_JWKS_URL, timeout=5)
            response.raise_for_status()
            _jwks_cache = response.json()
            _jwks_last_fetched = datetime.utcnow()
        except requests.RequestException as e:
            print(f"Error fetching JWKS: {e}")
            return None
    return _jwks_cache

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(
    subject: Union[str, Any], expires_delta: timedelta = None, extra_data: dict = None
) -> str:
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
    to_encode = {"exp": expire, "sub": str(subject)}
    if extra_data:
        to_encode.update(extra_data)
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

def decode_token(token: str) -> Optional[Dict[str, Any]]:
    """Intenta decodificar el token, comprobando si es externo o interno."""

    # Intento 1: Proveedor Externo (si JWKS está configurado)
    unverified_header = jwt.get_unverified_header(token)
    if "kid" in unverified_header and settings.EXTERNAL_AUTH_JWKS_URL:
        jwks = get_jwks()
        if jwks:

            # Formato RS256 para Auth0 / Clerk
            rsa_key = {}
            for key in jwks.get("keys", []):
                if key["kid"] == unverified_header["kid"]:
                    rsa_key = {
                        "kty": key["kty"],
                        "kid": key["kid"],
                        "use": key["use"],
                        "n": key["n"],
                        "e": key["e"]
                    }
                    break

            if rsa_key:
                try:
                    # En entornos reales suele usarse un audience o issuer
                    payload = jwt.decode(
                        token,
                        rsa_key,
                        algorithms=["RS256"],
                        options={"verify_aud": False, "verify_iss": False}
                    )
                    return payload
                except JWTError as e:
                    print(f"External JWT Error: {e}")
                    # Si falla, puede que intencionalmente fuera interno con 'kid'
                    pass

    # Intento 2: Custom JWT Interno (HS256)
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        return payload
    except JWTError as e:
        print(f"Internal JWT Error: {e}")
        return None
