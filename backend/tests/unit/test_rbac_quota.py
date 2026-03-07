"""
Tests unitarios para el sistema RBAC y Quota Limiting.

Prueba las dependencias de autorización y quota de forma aislada,
sin base de datos real ni red.

Ejecutar:
    cd backend
    uv run pytest tests/unit/test_rbac_quota.py -v
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

from app.models.user import User, UserRole, VALID_ROLES
from app.models.tenant import Tenant
from app.core.exceptions import QuotaExceededError, PermissionDeniedError


# ── UserRole ──────────────────────────────────────────────────────────────────

class TestUserRole:
    """Tests de los roles de usuario."""

    def test_valid_roles_contains_all(self):
        assert UserRole.ADMIN in VALID_ROLES
        assert UserRole.EDITOR in VALID_ROLES
        assert UserRole.VIEWER in VALID_ROLES

    def test_role_values(self):
        assert UserRole.ADMIN == "admin"
        assert UserRole.EDITOR == "editor"
        assert UserRole.VIEWER == "viewer"


# ── RBAC hierarchy ────────────────────────────────────────────────────────────

class TestRoleHierarchy:
    """Tests de la jerarquía de roles."""

    def test_hierarchy_import(self):
        from app.api.deps import _ROLE_HIERARCHY
        assert _ROLE_HIERARCHY[UserRole.ADMIN] > _ROLE_HIERARCHY[UserRole.EDITOR]
        assert _ROLE_HIERARCHY[UserRole.EDITOR] > _ROLE_HIERARCHY[UserRole.VIEWER]

    def test_admin_level_highest(self):
        from app.api.deps import _ROLE_HIERARCHY
        assert _ROLE_HIERARCHY[UserRole.ADMIN] == 3

    def test_viewer_level_lowest(self):
        from app.api.deps import _ROLE_HIERARCHY
        assert _ROLE_HIERARCHY[UserRole.VIEWER] == 1


# ── Quota Exceptions ─────────────────────────────────────────────────────────

class TestQuotaExceptions:
    """Tests del sistema de excepciones de cuota."""

    def test_quota_exceeded_error_attributes(self):
        exc = QuotaExceededError(
            resource="datasets",
            current=50,
            limit=50,
        )
        assert exc.status_code == 429
        assert exc.code == "QUOTA_EXCEEDED"
        assert "datasets" in exc.message
        assert "50/50" in exc.message

    def test_quota_exceeded_error_different_resources(self):
        exc = QuotaExceededError(resource="modelos", current=20, limit=20)
        assert "modelos" in exc.message

        exc = QuotaExceededError(resource="predicciones diarias", current=500, limit=500)
        assert "predicciones diarias" in exc.message

    def test_permission_denied_error(self):
        exc = PermissionDeniedError()
        assert exc.status_code == 403
        assert exc.code == "PERMISSION_DENIED"


# ── Tenant Quota Model ───────────────────────────────────────────────────────

class TestTenantQuotaModel:
    """Tests de los campos de cuota en el modelo Tenant."""

    def test_tenant_has_quota_fields(self):
        """El modelo Tenant debe tener los campos de cuota."""
        import inspect
        from app.models.tenant import Tenant
        # Check that the class has the columns
        assert hasattr(Tenant, "max_datasets")
        assert hasattr(Tenant, "max_models")
        assert hasattr(Tenant, "max_predictions_per_day")
        assert hasattr(Tenant, "max_training_jobs_per_day")


# ── Deps: require_role ────────────────────────────────────────────────────────

class TestRequireRole:
    """Tests de la function factory require_role."""

    def test_require_role_returns_callable(self):
        from app.api.deps import require_role
        checker = require_role("editor")
        assert callable(checker)

    def test_require_role_shorthands_exist(self):
        from app.api.deps import require_admin, require_editor, require_viewer
        assert callable(require_admin)
        assert callable(require_editor)
        assert callable(require_viewer)


# ── Deps: quota check functions ──────────────────────────────────────────────

class TestQuotaCheckFunctions:
    """Tests de las funciones de verificación de cuota."""

    def test_quota_check_functions_importable(self):
        from app.api.deps import (
            check_dataset_quota,
            check_model_quota,
            check_prediction_quota,
            check_training_quota,
        )
        assert callable(check_dataset_quota)
        assert callable(check_model_quota)
        assert callable(check_prediction_quota)
        assert callable(check_training_quota)


# ── Rate Limit module ────────────────────────────────────────────────────────

class TestRateLimitModule:
    """Tests del módulo de rate limiting."""

    def test_limiter_importable(self):
        from app.core.rate_limit import limiter
        assert limiter is not None

    def test_rate_limit_config_in_settings(self):
        from app.core.config import settings
        assert hasattr(settings, "RATE_LIMIT_TRAINING")
        assert hasattr(settings, "RATE_LIMIT_INFERENCE")
        # Verify format is something like "10/minute"
        assert "/" in settings.RATE_LIMIT_TRAINING
        assert "/" in settings.RATE_LIMIT_INFERENCE
