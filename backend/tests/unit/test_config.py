"""
Tests unitarios para la configuración (app.core.config).

Verifica los field_validators y la lógica de seguridad por entorno.

Ejecutar:
    cd backend
    uv run pytest tests/unit/test_config.py -v
"""

import pytest
import os


class TestSettingsValidators:
    """Tests de los validators de Settings."""

    def test_default_settings_load(self):
        """Los settings por defecto deben cargarse sin error en development."""
        from app.core.config import settings
        assert settings.PROJECT_NAME == "PraxisML"
        assert settings.ENVIRONMENT == "development"

    def test_rate_limit_format_valid(self):
        """Los rate limits por defecto tienen formato válido."""
        from app.core.config import settings
        assert "/" in settings.RATE_LIMIT_TRAINING
        assert "/" in settings.RATE_LIMIT_INFERENCE

    def test_rate_limit_format_validation(self):
        """Formato de rate limit inválido debe lanzar error."""
        from app.core.config import Settings
        with pytest.raises(Exception):  # ValidationError
            Settings(
                RATE_LIMIT_TRAINING="invalid_format",
                DATABASE_URL="postgresql://x:x@localhost/db",
            )

    def test_rate_limit_format_accepts_valid(self):
        """Formatos válidos de rate limit deben aceptarse."""
        from app.core.config import Settings
        for valid in ["5/second", "10/minute", "100/hour", "1000/day"]:
            s = Settings(
                RATE_LIMIT_TRAINING=valid,
                RATE_LIMIT_INFERENCE=valid,
                DATABASE_URL="postgresql://x:x@localhost/db",
            )
            assert s.RATE_LIMIT_TRAINING == valid

    def test_database_url_must_be_postgres(self):
        """DATABASE_URL debe comenzar con postgresql://."""
        from app.core.config import Settings
        with pytest.raises(Exception):
            Settings(DATABASE_URL="mysql://user:pass@localhost/db")

    def test_database_url_accepts_postgres(self):
        """DATABASE_URL con postgresql:// debe aceptarse."""
        from app.core.config import Settings
        s = Settings(DATABASE_URL="postgresql://user:pass@localhost/testdb")
        assert s.DATABASE_URL.startswith("postgresql://")

    def test_environment_validation(self):
        """ENVIRONMENT solo acepta valores válidos."""
        from app.core.config import Settings
        with pytest.raises(Exception):
            Settings(
                ENVIRONMENT="invalid_env",
                DATABASE_URL="postgresql://x:x@localhost/db",
            )

    def test_environment_accepts_valid(self):
        """Todos los entornos válidos deben aceptarse."""
        from app.core.config import Settings
        for env in ["development", "staging", "production", "testing"]:
            if env in ("staging", "production"):
                # Estos requieren SECRET_KEY no-default
                s = Settings(
                    ENVIRONMENT=env,
                    SECRET_KEY="a_very_secure_production_key_1234",
                    DATABASE_URL="postgresql://x:x@localhost/db",
                )
            else:
                s = Settings(
                    ENVIRONMENT=env,
                    DATABASE_URL="postgresql://x:x@localhost/db",
                )
            assert s.ENVIRONMENT == env

    def test_storage_backend_validation(self):
        """STORAGE_BACKEND solo acepta local, minio, s3."""
        from app.core.config import Settings
        with pytest.raises(Exception):
            Settings(
                STORAGE_BACKEND="gcs",
                DATABASE_URL="postgresql://x:x@localhost/db",
            )

    def test_storage_backend_accepts_valid(self):
        """Backends de storage válidos deben aceptarse."""
        from app.core.config import Settings
        for backend in ["local", "minio", "s3"]:
            s = Settings(
                STORAGE_BACKEND=backend,
                DATABASE_URL="postgresql://x:x@localhost/db",
            )
            assert s.STORAGE_BACKEND == backend

    def test_production_blocks_insecure_secret(self):
        """En production/staging, SECRET_KEY insegura debe lanzar error."""
        from app.core.config import Settings, _INSECURE_SECRET
        with pytest.raises(Exception) as exc_info:
            Settings(
                ENVIRONMENT="production",
                SECRET_KEY=_INSECURE_SECRET,
                DATABASE_URL="postgresql://x:x@localhost/db",
            )
        assert "SECRET_KEY" in str(exc_info.value)

    def test_production_allows_secure_secret(self):
        """En production, una SECRET_KEY segura debe funcionar."""
        from app.core.config import Settings
        s = Settings(
            ENVIRONMENT="production",
            SECRET_KEY="this_is_a_super_secure_256bit_key_for_prod",
            DATABASE_URL="postgresql://x:x@localhost/db",
        )
        assert s.ENVIRONMENT == "production"

    def test_cors_origins_list_property(self):
        """cors_origins_list debe parsear correctamente la string CSV."""
        from app.core.config import Settings
        s = Settings(
            CORS_ORIGINS="http://a.com, http://b.com , http://c.com",
            DATABASE_URL="postgresql://x:x@localhost/db",
        )
        assert s.cors_origins_list == ["http://a.com", "http://b.com", "http://c.com"]

    def test_cors_origins_list_empty(self):
        """cors_origins_list con string vacía debe devolver lista vacía."""
        from app.core.config import Settings
        s = Settings(
            CORS_ORIGINS="",
            DATABASE_URL="postgresql://x:x@localhost/db",
        )
        assert s.cors_origins_list == []

    def test_data_dir_is_absolute(self):
        """DATA_DIR por defecto debe ser un path absoluto."""
        from app.core.config import settings
        assert os.path.isabs(settings.DATA_DIR)

    def test_settings_singleton(self):
        """get_settings debe devolver la misma instancia (cached)."""
        from app.core.config import get_settings
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2
