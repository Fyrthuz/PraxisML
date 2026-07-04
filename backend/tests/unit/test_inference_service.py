from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


class TestInferenceService:
    def test_load_preprocessing_pipeline_none(self):
        from app.services.inference_service import InferenceService

        service = InferenceService()
        result = service.load_preprocessing_pipeline(None)
        assert result is None

    def test_preprocess_features_with_pipeline_none(self):
        from app.services.inference_service import InferenceService

        service = InferenceService()
        features = {"age": 25.0, "income": 50000.0}
        result = service.preprocess_features(features, feature_names=["age", "income"], pipeline=None)
        assert result is not None
        assert result.shape[1] == 2

    def test_preprocess_features_missing_features_defaults_zero(self):
        from app.services.inference_service import InferenceService

        service = InferenceService()
        features = {"age": 25.0}
        result = service.preprocess_features(features, feature_names=["age", "income"], pipeline=None)
        assert result is not None
        assert result.shape[1] == 2
        assert result[0][1] == 0.0

    def test_run_inference_sklearn_uncertainty(self):
        try:
            from sklearn.ensemble import RandomForestClassifier
            from app.core_ml.uncertainty.sklearn_uncertainty import entropy_uncertainty
        except ImportError:
            pytest.skip("sklearn_uncertainty no disponible")

        import numpy as np
        X_train = np.random.randn(20, 4)
        y_train = np.random.randint(0, 2, 20)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        X_test = np.random.randn(5, 4)
        probs = model.predict_proba(X_test)
        result = entropy_uncertainty(probs)
        assert result.shape == (5,)
        assert np.all(result >= 0)


class TestLoadModel:
    def test_load_model_with_torchscript(self, monkeypatch, tmp_path):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            pytest.skip("PyTorch no está instalado")

        from app.services.inference_service import InferenceService

        class MockMLModel:
            id = "model_ts_1"
            is_torchscript = True
            torchscript_path = str(tmp_path / "model.pt")
            mlflow_run_id = None

        (tmp_path / "model.pt").write_text("dummy")
        mock_model = MagicMock(spec=torch.nn.Module)
        monkeypatch.setattr("torch.jit.load", lambda path, map_location=None: mock_model)

        service = InferenceService(use_cache=False)
        result = service.load_model(MockMLModel(), device=torch.device("cpu"))
        assert result is mock_model

    def test_load_model_cache_hit(self, monkeypatch):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            pytest.skip("PyTorch no está instalado")

        from app.services.inference_service import InferenceService
        from app.services.model_cache import ModelCache

        class MockMLModel:
            id = "model_cache_1"
            is_torchscript = True
            torchscript_path = "/fake/path.pt"
            mlflow_run_id = None

        cache_key = "model_cache_1:True:/fake/path.pt"
        cached_model = MagicMock(spec=torch.nn.Module)

        test_cache = ModelCache(max_size=10)
        test_cache.set(cache_key, cached_model, ttl=3600)
        monkeypatch.setattr(
            "app.services.inference_service.get_model_cache",
            lambda: test_cache,
        )

        service = InferenceService(use_cache=True)
        result = service.load_model(MockMLModel(), device=torch.device("cpu"))
        assert result is cached_model

    def test_load_model_cache_miss_then_stores(self, monkeypatch, tmp_path):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            pytest.skip("PyTorch no está instalado")

        from app.services.inference_service import InferenceService
        from app.services.model_cache import ModelCache

        class MockMLModel:
            id = "model_miss_1"
            is_torchscript = True
            torchscript_path = str(tmp_path / "model.pt")
            mlflow_run_id = None

        (tmp_path / "model.pt").write_text("dummy")
        mock_model = MagicMock(spec=torch.nn.Module)
        monkeypatch.setattr("torch.jit.load", lambda path, map_location=None: mock_model)

        test_cache = ModelCache(max_size=10)
        monkeypatch.setattr(
            "app.services.inference_service.get_model_cache",
            lambda: test_cache,
        )

        service = InferenceService(use_cache=True)
        result = service.load_model(MockMLModel(), device=torch.device("cpu"))
        assert result is mock_model

        # Verify it's now in the cache
        cache_key = "model_miss_1:True:" + str(tmp_path / "model.pt")
        cached = test_cache.get(cache_key)
        assert cached is mock_model

    def test_load_model_torchscript_not_found(self):
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch no está instalado")

        from app.services.inference_service import InferenceService

        class MockMLModel:
            id = "model_nf"
            is_torchscript = True
            torchscript_path = "/nonexistent/model.pt"
            mlflow_run_id = None

        service = InferenceService(use_cache=False)
        with pytest.raises(FileNotFoundError):
            service.load_model(MockMLModel())


class TestPreprocessFeatures:
    def test_preprocess_features_with_pipeline(self):
        try:
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import StandardScaler
            from sklearn.compose import ColumnTransformer
            import pandas as pd
        except ImportError:
            pytest.skip("scikit-learn no está instalado")

        from app.services.inference_service import InferenceService

        pipeline = ColumnTransformer(
            transformers=[
                ("impute", SimpleImputer(strategy="mean"), ["age", "income"]),
                ("scale", StandardScaler(), ["age", "income"]),
            ],
            remainder="passthrough",
        )
        import numpy as np
        df = pd.DataFrame({"age": [25.0], "income": [50000.0]})
        pipeline.fit(df)

        service = InferenceService()
        features = {"age": 30.0, "income": 60000.0}
        result = service.preprocess_features(
            features, feature_names=["age", "income"], pipeline=pipeline,
        )
        assert result is not None
        assert result.shape[1] == 4
        assert result.dtype == np.float32

    def test_preprocess_features_extra_features(self):
        from app.services.inference_service import InferenceService

        features = {"age": 25.0, "income": 50000.0, "extra_col": "ignored"}
        service = InferenceService()
        result = service.preprocess_features(
            features, feature_names=["age", "income"], pipeline=None,
        )
        assert result is not None
        assert result.shape[1] == 2


class TestRunInference:
    def test_run_inference_with_numpy(self, monkeypatch):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            pytest.skip("PyTorch no está instalado")

        from app.services.inference_service import InferenceService
        import numpy as np

        model = nn.Linear(10, 2)
        input_data = np.random.randn(3, 10).astype(np.float32)

        mock_result = {"prediction": np.ones((3, 2)), "uncertainty": np.zeros(3)}
        mock_estimator = MagicMock()
        mock_estimator.estimate_uncertainty.return_value = mock_result

        monkeypatch.setattr(
            "app.core_ml.factory.PredictionFactory.get_estimator",
            lambda **kw: mock_estimator,
        )

        service = InferenceService(use_cache=False)
        result = service.run_inference(model, input_data, method="ensemble")
        assert result is mock_result
        assert "prediction" in result

    def test_run_inference_default_device(self, monkeypatch):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            pytest.skip("PyTorch no está instalado")

        from app.services.inference_service import InferenceService
        import numpy as np

        model = nn.Linear(10, 2)
        input_data = np.random.randn(2, 10).astype(np.float32)

        mock_result = {"prediction": np.ones((2, 2)), "uncertainty": np.zeros(2)}
        mock_estimator = MagicMock()
        mock_estimator.estimate_uncertainty.return_value = mock_result

        monkeypatch.setattr(
            "app.core_ml.factory.PredictionFactory.get_estimator",
            lambda **kw: mock_estimator,
        )

        service = InferenceService(use_cache=False)
        # Should use model's device (cpu) by default
        result = service.run_inference(model, input_data)
        assert result is mock_result
