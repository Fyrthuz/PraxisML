"""
Configuración centralizada de la aplicación.

Carga variables de entorno siguiendo el orden de precedencia:
  1. Variables de entorno del sistema / contenedor
  2. Archivo .env (raíz del proyecto o CWD)
  3. Defaults definidos aquí (solo válidos en development)

Validaciones de seguridad:
  - En production/staging, SECRET_KEY no puede ser el valor por defecto.
  - RATE_LIMIT_* debe tener formato "N/period" (e.g. "10/minute").
  - DATABASE_URL debe comenzar con "postgresql://".
"""

import os
import re
from pathlib import Path
from functools import lru_cache
from typing import List, Optional

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Constante de seguridad ─────────────────────────────────────────────────────
_INSECURE_SECRET = "super_secret_key_change_me_in_production"


class Settings(BaseSettings):
    """Configuración principal de la aplicación — leída desde variables de entorno."""

    # ── General ────────────────────────────────────────────────────────────────
    PROJECT_NAME: str = "PraxisML"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "development"  # development | staging | production

    # ── Base de datos ──────────────────────────────────────────────────────────
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/praxisml_db"

    # ── Almacenamiento local ───────────────────────────────────────────────────
    # Ruta base para ficheros en disco (datasets, modelos, predicciones).
    # Default: <proyecto>/backend/data
    DATA_DIR: str = str(Path(__file__).resolve().parent.parent.parent / "data")

    # ── CORS ───────────────────────────────────────────────────────────────────
    # Orígenes permitidos, separados por coma.
    # Ejemplo producción: "https://app.praxisml.ai,https://www.praxisml.ai"
    CORS_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000"

    @property
    def cors_origins_list(self) -> List[str]:
        """Devuelve la lista de orígenes CORS como List[str]."""
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    # ── Seguridad JWT ──────────────────────────────────────────────────────────
    SECRET_KEY: str = _INSECURE_SECRET
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 días

    # Proveedor externo opcional (Clerk / Auth0)
    EXTERNAL_AUTH_JWKS_URL: Optional[str] = None

    # ── Rate Limiting (slowapi) ────────────────────────────────────────────────
    # Formato: "N/period" donde period ∈ {second, minute, hour, day}
    RATE_LIMIT_TRAINING: str = "10/minute"
    RATE_LIMIT_INFERENCE: str = "30/minute"

    # ── Celery / Redis ─────────────────────────────────────────────────────────
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"

    # ── MLFlow ─────────────────────────────────────────────────────────────────
    MLFLOW_TRACKING_URI: str = ""

    # ── Storage Backend ────────────────────────────────────────────────────────
    # Valores: local | minio | s3
    STORAGE_BACKEND: str = "local"

    # MinIO (self-hosted, desarrollo/staging)
    MINIO_ENDPOINT: str = "http://minio:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET: str = "praxisml"
    MINIO_REGION: str = "us-east-1"

    # AWS S3 (producción cloud)
    S3_BUCKET: str = "praxisml-prod"
    AWS_DEFAULT_REGION: str = "eu-west-1"
    # AWS_ACCESS_KEY_ID y AWS_SECRET_ACCESS_KEY los lee boto3 automáticamente

    # ── Validators ─────────────────────────────────────────────────────────────

    @field_validator("DATABASE_URL")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        if not v.startswith("postgresql://") and not v.startswith("postgresql+asyncpg://"):
            raise ValueError(
                f"DATABASE_URL debe comenzar con 'postgresql://' — recibido: '{v[:30]}...'"
            )
        return v

    @field_validator("RATE_LIMIT_TRAINING", "RATE_LIMIT_INFERENCE")
    @classmethod
    def validate_rate_limit_format(cls, v: str) -> str:
        pattern = r"^\d+/(second|minute|hour|day)$"
        if not re.match(pattern, v):
            raise ValueError(
                f"Rate limit debe tener formato 'N/period' "
                f"(e.g. '10/minute') — recibido: '{v}'"
            )
        return v

    @field_validator("ENVIRONMENT")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        allowed = {"development", "staging", "production", "testing"}
        if v not in allowed:
            raise ValueError(
                f"ENVIRONMENT debe ser uno de {allowed} — recibido: '{v}'"
            )
        return v

    @field_validator("STORAGE_BACKEND")
    @classmethod
    def validate_storage_backend(cls, v: str) -> str:
        allowed = {"local", "minio", "s3"}
        if v not in allowed:
            raise ValueError(
                f"STORAGE_BACKEND debe ser uno de {allowed} — recibido: '{v}'"
            )
        return v

    @model_validator(mode="after")
    def check_production_security(self) -> "Settings":
        """En production/staging, la SECRET_KEY no puede ser el valor inseguro."""
        if self.ENVIRONMENT in ("production", "staging"):
            if self.SECRET_KEY == _INSECURE_SECRET:
                raise ValueError(
                    "⛔ SECRET_KEY insegura detectada en entorno "
                    f"'{self.ENVIRONMENT}'. Establece una clave segura de "
                    "al menos 32 caracteres mediante variable de entorno."
                )
        return self

    # ── Pydantic Settings Config ───────────────────────────────────────────────

    model_config = SettingsConfigDict(
        env_file=[
            os.path.join(
                os.path.dirname(__file__),  # app/core/
                "..", "..", "..", "..",      # → proyecto raíz
                ".env",
            ),
            ".env",  # fallback: CWD (útil en CI y Docker)
        ],
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    """Singleton cacheado de Settings — evita re-parsear .env en cada import."""
    return Settings()


settings = get_settings()
