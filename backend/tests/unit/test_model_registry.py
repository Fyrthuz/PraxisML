"""
Tests unitarios para el sistema de Model Registry y Versioning.

Ejecutar:
    cd backend
    uv run pytest tests/unit/test_model_registry.py -v
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

from app.models.ml_model import MLModel, ModelStage


class TestModelStage:
    """Tests para el enum de Stage."""

    def test_stage_values(self):
        assert ModelStage.STAGING == "Staging"
        assert ModelStage.PRODUCTION == "Production"
        assert ModelStage.ARCHIVED == "Archived"

    def test_stage_enum_has_three_values(self):
        assert len(ModelStage) == 3


class TestMLModelVersioning:
    """Tests para los campos de versionado en MLModel."""

    def test_model_has_version_field(self):
        assert hasattr(MLModel, "version")

    def test_model_has_stage_field(self):
        assert hasattr(MLModel, "stage")

    def test_model_has_promoted_at_field(self):
        assert hasattr(MLModel, "promoted_at")

    def test_model_has_promoted_by_field(self):
        assert hasattr(MLModel, "promoted_by")

    def test_model_has_mlflow_registry_name_field(self):
        assert hasattr(MLModel, "mlflow_registry_name")

    def test_model_has_mlflow_version_field(self):
        assert hasattr(MLModel, "mlflow_version")


class TestMLFlowServiceRegistry:
    """Tests para los metodos de MLflow Registry en MLFlowService."""

    def test_mlflow_service_has_registry_methods(self):
        """Verify MLFlowService has registry methods."""
        from app.services.mlflow_service import MLFlowService

        assert hasattr(MLFlowService, "register_model_to_registry")
        assert hasattr(MLFlowService, "transition_model_stage")
        assert hasattr(MLFlowService, "get_model_versions")
        assert hasattr(MLFlowService, "get_registered_models")


class TestDVCService:
    """Tests para el DVCService."""

    @patch("subprocess.run")
    @patch("pathlib.Path.mkdir")
    def test_init_repository(self, mock_mkdir, mock_run):
        from app.services.dvc_service import DVCService
        
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        
        service = DVCService("test-tenant")
        # Mocking existence of .dvc to avoid actual init if it exists
        with patch("pathlib.Path.exists", return_value=False):
            result = service.init_repository()
            assert result is True
            mock_run.assert_called()

    @patch("subprocess.run")
    @patch("pathlib.Path.mkdir")
    def test_get_file_hash(self, mock_mkdir, mock_run):
        from app.services.dvc_service import DVCService
        import tempfile
        import os

        service = DVCService("test-tenant")
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(b"hello world")
            temp_path = tf.name

        try:
            result = service.get_file_hash(temp_path)
            # md5 of "hello world"
            assert result == "5eb63bbbe01eeed093cb22bb8f5acdc3"
        finally:
            os.unlink(temp_path)

    def test_dvc_service_imports(self):
        """Verify DVC service functions are importable."""
        from app.services.dvc_service import get_dvc_service, track_dataset_with_dvc

        assert callable(get_dvc_service)
        assert callable(track_dataset_with_dvc)
