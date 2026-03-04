import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    PROJECT_NAME: str = "Antigravity SaaS"
    API_V1_STR: str = "/api/v1"

    # Base de datos (PostgreSQL)
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/antigravity_db"

    # Almacenamiento local (MVP)
    DATA_DIR: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data"
    )

    # Security
    SECRET_KEY: str = "super_secret_key_change_me_in_production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7 # 7 days
    
    # External Auth Provider
    EXTERNAL_AUTH_JWKS_URL: str | None = None # e.g. "https://YOUR_DOMAIN/.well-known/jwks.json"

    # Celery / Redis
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"

    # MLFlow (puede apuntar a un servidor remoto en producción)
    # Por defecto usa almacenamiento local de archivos
    MLFLOW_TRACKING_URI: str = ""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()
