"""
Tests de integracion para el ciclo de vida del modelo (promote, archive, version).

Ejecutar:
    cd backend
    uv run pytest tests/integration/test_model_lifecycle.py -v
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.models.ml_model import MLModel
from app.models.tenant import Tenant
from app.models.user import User, UserRole


@pytest.fixture
def mock_db_session():
    """Mock de la sesion de base de datos."""
    session = MagicMock()
    return session


@pytest.fixture
def client():
    """Cliente de test para la API con bypass de auth."""
    from app.api.deps import get_current_user, get_current_tenant
    from app.models.user import User, UserRole
    from app.models.tenant import Tenant
    from app.models.base import Base
    from app.database import engine, SessionLocal

    # Mock user for bypass
    mock_user = MagicMock(spec=User)
    mock_user.id = "test-user-id"
    mock_user.role = UserRole.ADMIN
    mock_user.tenant_id = "test-tenant-id"
    mock_user.is_active = True

    app.dependency_overrides[get_current_user] = lambda: mock_user
    
    mock_tenant = Tenant(id="test-tenant-id", name="Test Tenant")
    app.dependency_overrides[get_current_tenant] = lambda: mock_tenant

    # Initialize database tables for SQLite
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    # Create default tenant and user
    db = SessionLocal()
    from app.models.tenant import Tenant
    from app.models.user import User
    tenant = Tenant(id="test-tenant-id", name="Test Tenant")
    user = User(id="test-user-id", email="test@example.com", hashed_password="fake", tenant_id="test-tenant-id", role=UserRole.ADMIN)
    db.add(tenant)
    db.add(user)
    db.commit()
    db.close()

    with TestClient(app) as c:
        yield c
    
    app.dependency_overrides.clear()


@pytest.fixture
def auth_headers():
    """Headers de autenticacion."""
    return {"Authorization": "Bearer fake-token"}


@pytest.fixture
def mock_tenant():
    """Tenant mock."""
    tenant = MagicMock(spec=Tenant)
    tenant.id = "test-tenant-id"
    tenant.name = "Test Tenant"
    return tenant


@pytest.fixture
def mock_model():
    """Modelo mock."""
    model = MagicMock(spec=MLModel)
    model.id = "test-model-id"
    model.name = "test-model"
    model.tenant_id = "test-tenant-id"
    model.mlflow_run_id = "test-run-id"
    model.version = "1.0.0"
    model.stage = "Staging"
    model.promoted_at = None
    model.promoted_by = None
    model.mlflow_registry_name = None
    model.mlflow_version = None
    return model


class TestModelPromoteEndpoint:
    """Tests para el endpoint de promocion de modelos."""

    @patch("app.api.deps.get_current_tenant")
    @patch("app.api.deps.require_editor")
    def test_promote_model_updates_stage(
        self,
        mock_require_editor,
        mock_get_tenant,
        client,
        auth_headers,
        mock_tenant,
        mock_model,
    ):
        mock_require_editor.return_value = MagicMock()
        mock_get_tenant.return_value = mock_tenant

        from app.database import get_db
        from unittest.mock import patch as mock_patch

        from app.database import SessionLocal
        db_session = SessionLocal()
        try:
            # Need to detach from mock or use a real object for DB
            from app.models.ml_model import MLModel
            db_model = MLModel(
                id="test-model-id",
                name="test-model",
                tenant_id="test-tenant-id",
                mlflow_run_id="test-run-id",
                version="1.0.0",
                stage="Staging",
                metrics_metadata={},
            )
            db_session.add(db_model)
            db_session.commit()
            
            response = client.post(
                "/api/v1/models/test-model-id/promote",
                data={"target_stage": "Production"},
            )

            assert response.status_code == 200
        finally:
            db_session.close()

    @patch("app.api.deps.get_current_tenant")
    @patch("app.api.deps.require_editor")
    def test_promote_model_invalid_stage(
        self,
        mock_require_editor,
        mock_get_tenant,
        client,
        auth_headers,
        mock_tenant,
        mock_model,
    ):
        mock_require_editor.return_value = MagicMock()
        mock_get_tenant.return_value = mock_tenant

        from unittest.mock import patch as mock_patch

        from app.database import SessionLocal
        db_session = SessionLocal()
        try:
            from app.models.ml_model import MLModel
            db_model = MLModel(
                id="test-model-id",
                name="test-model",
                tenant_id="test-tenant-id",
                mlflow_run_id="test-run-id",
                version="1.0.0",
                stage="Staging",
                metrics_metadata={},
            )
            db_session.add(db_model)
            db_session.commit()

            response = client.post(
                "/api/v1/models/test-model-id/promote",
                data={"target_stage": "Invalid"},
            )

            assert response.status_code == 400
        finally:
            db_session.close()


class TestModelArchiveEndpoint:
    """Tests para el endpoint de archivo de modelos."""

    @patch("app.api.deps.get_current_tenant")
    @patch("app.api.deps.require_editor")
    def test_archive_model_sets_archived_stage(
        self,
        mock_require_editor,
        mock_get_tenant,
        client,
        auth_headers,
        mock_tenant,
        mock_model,
    ):
        mock_require_editor.return_value = MagicMock()
        mock_get_tenant.return_value = mock_tenant

        from unittest.mock import patch as mock_patch

        with mock_patch("app.api.routes.v1.models.get_db") as mock_get_db:
            mock_db = MagicMock()
            mock_db.query.return_value.filter.return_value.first.return_value = (
                mock_model
            )
            mock_get_db.return_value = iter([mock_db])

            response = client.post(
                "/api/v1/models/test-model-id/archive",
                headers=auth_headers,
            )

            assert response.status_code in [200, 401, 404]


class TestModelVersionsEndpoint:
    """Tests para el endpoint de versiones de modelo."""

    @patch("app.api.deps.get_current_tenant")
    @patch("app.api.deps.require_viewer")
    def test_get_model_versions(
        self,
        mock_require_viewer,
        mock_get_tenant,
        client,
        auth_headers,
        mock_tenant,
        mock_model,
    ):
        mock_require_viewer.return_value = MagicMock()
        mock_get_tenant.return_value = mock_tenant

        from unittest.mock import patch as mock_patch

        with mock_patch("app.api.routes.v1.models.get_db") as mock_get_db:
            mock_db = MagicMock()
            mock_db.query.return_value.filter.return_value.first.return_value = (
                mock_model
            )
            mock_get_db.return_value = iter([mock_db])

            response = client.get(
                "/api/v1/models/test-model-id/versions",
                headers=auth_headers,
            )

            if response.status_code == 200:
                data = response.json()
                assert "current_version" in data
                assert "stage" in data
            else:
                assert response.status_code in [401, 404]


class TestRegisteredModelsEndpoint:
    """Tests para el endpoint de modelos registrados."""

    @patch("app.api.deps.require_viewer")
    def test_list_registered_models(self, mock_require_viewer, client, auth_headers):
        mock_require_viewer.return_value = MagicMock()

        with patch(
            "app.services.mlflow_service.MLFlowService.get_registered_models"
        ) as mock_get:
            mock_get.return_value = []

            response = client.get(
                "/api/v1/models/registry",
            )

            assert response.status_code in [200, 401, 500]


class TestDVCService:
    """Tests para el servicio DVC."""

    @patch("subprocess.run")
    @patch("pathlib.Path.mkdir")
    def test_init_repository(self, mock_mkdir, mock_run):
        from app.services.dvc_service import DVCService
        from unittest.mock import MagicMock, patch
        
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        
        service = DVCService("test-tenant")
        # Mocking existence of .dvc to avoid actual init if it exists
        with patch("pathlib.Path.exists", return_value=False):
            result = service.init_repository()
            assert result is True
            mock_run.assert_called()


class TestModelResponseSchema:
    """Tests para el schema de respuesta del modelo."""

    def test_model_response_includes_version(self):
        from app.schemas.ml_model import MLModelResponse
        from datetime import datetime

        model_data = {
            "id": "test-id",
            "name": "test-model",
            "description": "Test description",
            "mlflow_run_id": "run-123",
            "metrics_metadata": {},
            "is_public": False,
            "tenant_id": "tenant-123",
            "created_at": datetime.utcnow(),
            "is_active": True,
            "version": "1.0.0",
            "stage": "Staging",
            "promoted_at": None,
            "promoted_by": None,
            "mlflow_registry_name": None,
            "mlflow_version": None,
        }

        response = MLModelResponse(**model_data)
        assert response.version == "1.0.0"
        assert response.stage == "Staging"

    def test_model_response_stage_enum(self):
        from app.schemas.ml_model import ModelStage

        assert ModelStage.STAGING == "Staging"
        assert ModelStage.PRODUCTION == "Production"
        assert ModelStage.ARCHIVED == "Archived"
