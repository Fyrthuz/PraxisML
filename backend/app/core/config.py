import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import List


class Settings(BaseSettings):
    PROJECT_NAME: str = "Antigravity SaaS"
    API_V1_STR: str = "/api/v1"

    # Entorno: development | staging | production
    ENVIRONMENT: str = "development"

    # Base de datos (PostgreSQL)
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/antigravity_db"

    # Almacenamiento local (usado cuando STORAGE_BACKEND=local)
    DATA_DIR: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data"
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    # Lista de orígenes permitidos separados por coma.
    # Ejemplo producción: "https://app.antigravity.ai,https://www.antigravity.ai"
    # En development se definen orígenes localhost por defecto.
    CORS_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000"

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    # ── Security ───────────────────────────────────────────────────────────────
    SECRET_KEY: str = "super_secret_key_change_me_in_production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days

    # External Auth Provider
    EXTERNAL_AUTH_JWKS_URL: str | None = None  # e.g. "https://YOUR_DOMAIN/.well-known/jwks.json"

    # ── Rate Limiting (slowapi) ────────────────────────────────────────────────
    # Límite global por IP para endpoints de entrenamiento/inferencia
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

    # MinIO (self-hosted)
    MINIO_ENDPOINT: str = "http://minio:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET: str = "antigravity"
    MINIO_REGION: str = "us-east-1"

    # AWS S3 (producción cloud)
    S3_BUCKET: str = "antigravity-prod"
    AWS_DEFAULT_REGION: str = "eu-west-1"
    # AWS_ACCESS_KEY_ID y AWS_SECRET_ACCESS_KEY los lee boto3 automáticamente

    model_config = SettingsConfigDict(
        # Busca .env en la raíz del proyecto (cuatro niveles arriba de este archivo)
        # Orden de búsqueda: raíz del proyecto → directorio de trabajo actual
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
def get_settings():
    return Settings()


settings = get_settings()
