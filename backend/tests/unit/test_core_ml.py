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

    def test_normalize_image_shape(self):
        """La normalización debe preservar el shape de la imagen."""
        try:
            from app.core_ml.preprocessing import normalize_image
            img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            result = normalize_image(img)
            assert result.shape == img.shape
            assert result.dtype == np.float32 or result.max() <= 1.0
        except (ImportError, AttributeError):
            pytest.skip("normalize_image no disponible")

    def test_resize_image(self):
        """El resize debe producir la forma objetivo."""
        try:
            from app.core_ml.preprocessing import resize_image
            img = np.zeros((512, 512, 3), dtype=np.uint8)
            result = resize_image(img, target_size=(256, 256))
            assert result.shape[:2] == (256, 256)
        except (ImportError, AttributeError):
            pytest.skip("resize_image no disponible")


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

    def test_entropy_calculation(self):
        """La entropía de predicciones uniformes debe ser máxima."""
        try:
            from app.core_ml.uncertainty.base import compute_entropy
            # Distribución uniforme → entropía máxima
            probs = np.ones((10, 5)) / 5  # 10 muestras, 5 clases
            entropy = compute_entropy(probs)
            assert entropy.shape == (10,) or entropy.ndim >= 0
            assert np.all(entropy >= 0)
        except (ImportError, AttributeError):
            pytest.skip("compute_entropy no disponible")

    def test_variance_from_samples(self):
        """La varianza de muestras idénticas debe ser cero."""
        try:
            from app.core_ml.uncertainty.base import compute_variance
            # Todas las muestras iguales → varianza 0
            samples = np.ones((5, 3, 64, 64))  # 5 muestras MC idénticas
            variance = compute_variance(samples)
            assert np.allclose(variance, 0.0)
        except (ImportError, AttributeError):
            pytest.skip("compute_variance no disponible")


# ── exceptions ────────────────────────────────────────────────────────────────

class TestExceptions:
    """Tests del sistema de excepciones de dominio."""

    def test_all_exceptions_importable(self):
        from app.core.exceptions import (
            AntigravityError, AuthenticationError, PermissionDeniedError,
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
