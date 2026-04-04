from datetime import datetime, timedelta, timezone
from typing import Any, Union, Dict, Optional
from jose import jwt, JWTError
from passlib.context import CryptContext
import requests
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Cache para las JWKS de proveedores externos (simplificado)
_jwks_cache = None
_jwks_last_fetched = None


def get_jwks() -> Optional[Dict]:
    global _jwks_cache, _jwks_last_fetched

    if not settings.EXTERNAL_AUTH_JWKS_URL:
        return None

    # Recargar cada hora la cache de JWKS
    if (
        _jwks_cache is None
        or _jwks_last_fetched is None
        or (datetime.now(timezone.utc) - _jwks_last_fetched) > timedelta(hours=1)
    ):
        try:
            response = requests.get(settings.EXTERNAL_AUTH_JWKS_URL, timeout=5)
            response.raise_for_status()
            _jwks_cache = response.json()
            _jwks_last_fetched = datetime.now(timezone.utc)
        except requests.RequestException as e:
            logger.error("Error fetching JWKS: %s", str(e))
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
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
    to_encode = {"exp": expire, "sub": str(subject)}
    if extra_data:
        to_encode.update(extra_data)
    encoded_jwt = jwt.encode(
        to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM
    )
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
                        "e": key["e"],
                    }
                    break

            if rsa_key:
                decode_options = {"verify_aud": True, "verify_iss": True}
                decode_kwargs = {
                    "token": token,
                    "key": rsa_key,
                    "algorithms": ["RS256"],
                    "options": decode_options,
                }
                if settings.EXTERNAL_AUTH_AUDIENCE:
                    decode_kwargs["audience"] = settings.EXTERNAL_AUTH_AUDIENCE
                if settings.EXTERNAL_AUTH_ISSUER:
                    decode_kwargs["issuer"] = settings.EXTERNAL_AUTH_ISSUER

                try:
                    payload = jwt.decode(**decode_kwargs)
                    return payload
                except JWTError as e:
                    logger.warning("External JWT Error: %s", str(e))
                    pass

    # Intento 2: Custom JWT Interno (HS256)
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        return payload
    except JWTError as e:
        logger.warning("Internal JWT Error: %s", str(e))
        return None
