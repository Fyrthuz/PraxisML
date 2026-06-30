"""
Tests unitarios del módulo core_ml.

Prueba las funciones de factory, preprocessing y los estimadores
de incertidumbre de forma aislada (sin base de datos ni red).

Ejecutar:
    cd backend
    uv run pytest tests/unit/ -v
"""

import pytest
import numpy as np


# ── factory.py ────────────────────────────────────────────────────────────────

class TestFactory:
    """Tests del factory de modelos ML."""

    def test_factory_import(self):
        """El módulo factory debe importarse sin errores."""
        from app.core_ml import factory
        assert factory is not None

    def test_get_model_raises_for_unknown_type(self):
        """Un tipo de modelo desconocido debe lanzar excepción."""
        from app.core_ml.factory import PredictionFactory, UncertaintyMethod
        import pytest
        with pytest.raises(Exception):
            # Pasar un método inválido debe fallar
            PredictionFactory(method="tipo_invalido_xyz")

    def test_get_mlp_model(self):
        """El factory debe exponer el método get_estimator."""
        try:
            from app.core_ml.factory import PredictionFactory, UncertaintyMethod
            # Verificamos que get_estimator es callable (método estático)
            assert callable(getattr(PredictionFactory, "get_estimator", None))
        except ImportError:
            pytest.skip("Dependencias no disponibles en este entorno")


# ── preprocessing.py ──────────────────────────────────────────────────────────

class TestPreprocessing:
    """Tests del módulo de preprocesamiento."""

    def test_preprocessing_import(self):
        from app.core_ml import preprocessing
        assert preprocessing is not None

    def test_build_pipeline(self):
        """Construir pipeline con pasos básicos de imputación y escalado."""
        try:
            from app.core_ml.preprocessing import build_pipeline
            config = {
                "steps": [
                    {"type": "impute", "columns": ["age", "income"], "strategy": "median"},
                    {"type": "scale", "columns": ["age", "income"], "strategy": "standard"},
                ]
            }
            pipeline = build_pipeline(config, column_names=["age", "income", "target"])
            assert pipeline is not None
        except (ImportError, AttributeError):
            pytest.skip("build_pipeline no disponible")

    def test_apply_pipeline(self):
        """Aplicar pipeline transforma el DataFrame correctamente."""
        try:
            from app.core_ml.preprocessing import build_pipeline, apply_pipeline
            import pandas as pd
            df = pd.DataFrame({"age": [1.0, 2.0, 3.0], "income": [100.0, 200.0, 300.0], "target": [0, 1, 0]})
            config = {
                "steps": [
                    {"type": "impute", "columns": ["age", "income"], "strategy": "median"},
                ]
            }
            pipeline = build_pipeline(config, column_names=["age", "income", "target"])
            X, y = apply_pipeline(pipeline, df, target_column="target")
            assert X.shape[1] == 2
            assert len(y) == 3
        except (ImportError, AttributeError):
            pytest.skip("apply_pipeline no disponible")


# ── uncertainty ───────────────────────────────────────────────────────────────

class TestUncertaintyEstimators:
    """Tests de los estimadores de incertidumbre."""

    def test_uncertainty_module_import(self):
        from app.core_ml import uncertainty
        assert uncertainty is not None

    def test_mc_dropout_estimator_import(self):
        try:
            from app.core_ml.uncertainty import mc_dropout
            assert mc_dropout is not None
        except ImportError:
            pytest.skip("mc_dropout no disponible")

    def test_base_uncertainty_has_entropy(self):
        """BaseUncertaintyEstimator debe exponer _compute_entropy."""
        try:
            import torch
            from app.core_ml.uncertainty.base import BaseUncertaintyEstimator

            batch = torch.ones(10, 5) / 5
            entropy = BaseUncertaintyEstimator._compute_entropy(batch)
            assert entropy.shape == (10,)
            assert torch.all(entropy >= 0)
        except (ImportError, AttributeError):
            pytest.skip("BaseUncertaintyEstimator no disponible")


# ── exceptions ────────────────────────────────────────────────────────────────

class TestExceptions:
    """Tests del sistema de excepciones de dominio."""

    def test_all_exceptions_importable(self):
        from app.core.exceptions import (
            PraxisMLError, AuthenticationError, PermissionDeniedError,
            ModelNotFoundError, TrainingError, StorageError,
            InferenceError, DatasetNotFoundError, StorageObjectNotFoundError,
            QuotaExceededError,
        )
        assert ModelNotFoundError is not None
        assert QuotaExceededError is not None

    def test_model_not_found_error_attributes(self):
        from app.core.exceptions import ModelNotFoundError
        exc = ModelNotFoundError(model_id="abc-123")
        assert exc.status_code == 404
        assert exc.code == "MODEL_NOT_FOUND"
        assert "abc-123" in exc.message

    def test_training_error_attributes(self):
        from app.core.exceptions import TrainingError
        exc = TrainingError("Fallo al entrenar", detail={"epoch": 5})
        assert exc.status_code == 500
        assert exc.code == "TRAINING_ERROR"
        assert exc.detail == {"epoch": 5}

    def test_storage_error_attributes(self):
        from app.core.exceptions import StorageError
        exc = StorageError()
        assert exc.status_code == 500
        assert exc.code == "STORAGE_ERROR"

    def test_permission_denied_error(self):
        from app.core.exceptions import PermissionDeniedError
        exc = PermissionDeniedError()
        assert exc.status_code == 403


# ── StorageService ─────────────────────────────────────────────────────────────

class TestLocalStorageService:
    """Tests del StorageService con backend local (sin MinIO)."""

    def test_upload_and_download(self, tmp_path, monkeypatch):
        """Upload y download deben ser inversos."""
        import os
        monkeypatch.setenv("STORAGE_BACKEND", "local")

        # Parchear DATA_DIR para usar tmp_path
        from app.core import config as cfg
        original_data_dir = cfg.settings.DATA_DIR
        cfg.settings.__dict__["DATA_DIR"] = str(tmp_path)

        try:
            from app.services.storage_local import LocalStorageService
            storage = LocalStorageService.__new__(LocalStorageService)
            storage.base_dir = tmp_path / "storage"
            storage.base_dir.mkdir(parents=True, exist_ok=True)

            key = "tenant1/predictions/test.npy"
            data = b"\x93NUMPY"  # magic bytes de un .npy
            storage.upload(key, data)
            result = storage.download(key)
            assert result == data
        finally:
            cfg.settings.__dict__["DATA_DIR"] = original_data_dir

    def test_exists_false_for_missing(self, tmp_path):
        from app.services.storage_local import LocalStorageService
        storage = LocalStorageService.__new__(LocalStorageService)
        storage.base_dir = tmp_path / "storage"
        storage.base_dir.mkdir(parents=True, exist_ok=True)

        assert not storage.exists("no/existe/este.npy")

    def test_download_missing_raises(self, tmp_path):
        from app.services.storage_local import LocalStorageService
        from app.core.exceptions import StorageObjectNotFoundError
        storage = LocalStorageService.__new__(LocalStorageService)
        storage.base_dir = tmp_path / "storage"
        storage.base_dir.mkdir(parents=True, exist_ok=True)

        with pytest.raises(StorageObjectNotFoundError):
            storage.download("no/existe/este.npy")
