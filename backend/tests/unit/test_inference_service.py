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
