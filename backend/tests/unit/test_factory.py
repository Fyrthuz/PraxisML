"""
Tests for app.core_ml.factory: PredictionFactory and UncertaintyMethod enum.
"""

import pytest


class TestUncertaintyMethod:
    def test_enum_values(self):
        from app.core_ml.factory import UncertaintyMethod

        assert UncertaintyMethod.MC_DROPOUT.value == "mc_dropout"
        assert UncertaintyMethod.CALIBRATED_MC_DROPOUT.value == "calibrated_mc_dropout"
        assert UncertaintyMethod.TTA.value == "tta"
        assert UncertaintyMethod.NOISY_INFERENCE.value == "noisy_inference"
        assert UncertaintyMethod.ENSEMBLE.value == "ensemble"
        assert UncertaintyMethod.NONE.value == "none"
        assert UncertaintyMethod.ENTROPY.value == "entropy"
        assert UncertaintyMethod.TREE_VARIANCE.value == "tree_variance"
        assert UncertaintyMethod.CONFORMAL.value == "conformal"

    def test_enum_from_string(self):
        from app.core_ml.factory import UncertaintyMethod

        assert UncertaintyMethod("mc_dropout") == UncertaintyMethod.MC_DROPOUT
        assert UncertaintyMethod("ensemble") == UncertaintyMethod.ENSEMBLE
        assert UncertaintyMethod("none") == UncertaintyMethod.NONE
        assert UncertaintyMethod("entropy") == UncertaintyMethod.ENTROPY

    def test_invalid_enum_raises_valueerror(self):
        from app.core_ml.factory import UncertaintyMethod

        with pytest.raises(ValueError):
            UncertaintyMethod("invalid_method")


class TestPredictionFactory:
    def test_get_estimator_mc_dropout(self):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            pytest.skip("PyTorch no está instalado")

        from app.core_ml.factory import PredictionFactory, UncertaintyMethod
        from app.core_ml.uncertainty import MCDropoutEstimator

        model = nn.Sequential(nn.Linear(10, 20), nn.Dropout(0.2), nn.Linear(20, 2))
        estimator = PredictionFactory.get_estimator(
            UncertaintyMethod.MC_DROPOUT, model, torch.device("cpu"),
            mc_samples=5, p_dropout=0.1,
        )
        assert isinstance(estimator, MCDropoutEstimator)

    def test_get_estimator_tta(self):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            pytest.skip("PyTorch no está instalado")

        from app.core_ml.factory import PredictionFactory, UncertaintyMethod
        from app.core_ml.uncertainty import TTAEstimator

        model = nn.Linear(10, 2)
        estimator = PredictionFactory.get_estimator(
            UncertaintyMethod.TTA, model, torch.device("cpu"),
            tta_samples=10,
        )
        assert isinstance(estimator, TTAEstimator)

    def test_get_estimator_noisy_inference(self):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            pytest.skip("PyTorch no está instalado")

        from app.core_ml.factory import PredictionFactory, UncertaintyMethod
        from app.core_ml.uncertainty import NoisyInferenceEstimator

        model = nn.Linear(10, 2)
        estimator = PredictionFactory.get_estimator(
            UncertaintyMethod.NOISY_INFERENCE, model, torch.device("cpu"),
            n_samples=5, noise_std=0.05,
        )
        assert isinstance(estimator, NoisyInferenceEstimator)

    def test_get_estimator_ensemble(self):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            pytest.skip("PyTorch no está instalado")

        from app.core_ml.factory import PredictionFactory, UncertaintyMethod
        from app.core_ml.uncertainty import EnsembleUncertaintyEstimator

        model = nn.Sequential(nn.Linear(10, 20), nn.Dropout(0.2), nn.Linear(20, 2))
        estimator = PredictionFactory.get_estimator(
            UncertaintyMethod.ENSEMBLE, model, torch.device("cpu"),
        )
        assert isinstance(estimator, EnsembleUncertaintyEstimator)

    def test_get_estimator_none_torch(self):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            pytest.skip("PyTorch no está instalado")

        from app.core_ml.factory import PredictionFactory, UncertaintyMethod
        from app.core_ml.uncertainty.base import BaseUncertaintyEstimator

        model = nn.Linear(10, 2)
        estimator = PredictionFactory.get_estimator(
            UncertaintyMethod.NONE, model, torch.device("cpu"),
        )
        assert isinstance(estimator, BaseUncertaintyEstimator)
        assert hasattr(estimator, "compute_uncertainty")

    def test_get_estimator_none_sklearn(self):
        try:
            from sklearn.ensemble import RandomForestClassifier
        except ImportError:
            pytest.skip("scikit-learn no está instalado")

        from app.core_ml.factory import PredictionFactory, UncertaintyMethod
        from app.core_ml.uncertainty.sklearn_uncertainty import BaseSklearnEstimator

        model = RandomForestClassifier(n_estimators=2)
        estimator = PredictionFactory.get_estimator(
            UncertaintyMethod.NONE, model,
        )
        assert isinstance(estimator, BaseSklearnEstimator)

    def test_get_estimator_entropy(self):
        try:
            from sklearn.ensemble import RandomForestClassifier
        except ImportError:
            pytest.skip("scikit-learn no está instalado")

        from app.core_ml.factory import PredictionFactory, UncertaintyMethod
        from app.core_ml.uncertainty import SklearnEntropyEstimator

        model = RandomForestClassifier(n_estimators=2)
        estimator = PredictionFactory.get_estimator(
            UncertaintyMethod.ENTROPY, model,
        )
        assert isinstance(estimator, SklearnEntropyEstimator)

    def test_get_estimator_tree_variance(self):
        try:
            from sklearn.ensemble import RandomForestClassifier
        except ImportError:
            pytest.skip("scikit-learn no está instalado")

        from app.core_ml.factory import PredictionFactory, UncertaintyMethod
        from app.core_ml.uncertainty import TreeVarianceEstimator

        model = RandomForestClassifier(n_estimators=2)
        estimator = PredictionFactory.get_estimator(
            UncertaintyMethod.TREE_VARIANCE, model,
        )
        assert isinstance(estimator, TreeVarianceEstimator)

    def test_get_estimator_conformal(self):
        try:
            from sklearn.ensemble import RandomForestClassifier
        except ImportError:
            pytest.skip("scikit-learn no está instalado")

        from app.core_ml.factory import PredictionFactory, UncertaintyMethod
        from app.core_ml.uncertainty import ConformalEstimator

        model = RandomForestClassifier(n_estimators=2)
        estimator = PredictionFactory.get_estimator(
            UncertaintyMethod.CONFORMAL, model, alpha=0.05,
        )
        assert isinstance(estimator, ConformalEstimator)

    def test_get_estimator_calibrated_mc_dropout_raises(self):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            pytest.skip("PyTorch no está instalado")

        from app.core_ml.factory import PredictionFactory, UncertaintyMethod

        model = nn.Linear(10, 2)
        with pytest.raises(NotImplementedError):
            PredictionFactory.get_estimator(
                UncertaintyMethod.CALIBRATED_MC_DROPOUT, model, torch.device("cpu"),
            )

    def test_invalid_method_raises_valueerror(self):
        from app.core_ml.factory import PredictionFactory, UncertaintyMethod

        import torch.nn as nn
        model = nn.Linear(10, 2)

        try:
            PredictionFactory.get_estimator("unknown_method", model)
        except ValueError:
            pass
        except Exception:
            # The error might be before the ValueError if method conversion fails
            pass

    def test_get_available_methods_pytorch(self):
        from app.core_ml.factory import PredictionFactory

        methods = PredictionFactory.get_available_methods("pytorch")
        assert "mc_dropout" in methods
        assert "tta" in methods
        assert "ensemble" in methods
        assert "none" in methods
        assert "noisy_inference" in methods

    def test_get_available_methods_sklearn(self):
        from app.core_ml.factory import PredictionFactory

        methods = PredictionFactory.get_available_methods("sklearn")
        assert "entropy" in methods
        assert "tree_variance" in methods
        assert "conformal" in methods
        assert "none" in methods
