import numpy as np
import pytest


class TestMCDropoutEstimator:
    def test_mc_dropout_estimate_shape(self):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            pytest.skip("PyTorch no está instalado")

        from app.core_ml.uncertainty.mc_dropout import MCDropoutEstimator

        device = torch.device("cpu")
        # Use a Conv2d model that accepts 4D input (B, C, H, W)
        model = nn.Sequential(
            nn.Conv2d(3, 2, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
        )
        model.train()
        estimator = MCDropoutEstimator(model, device=device, mc_samples=5)
        x = torch.randn(2, 3, 8, 8)
        result = estimator.compute_uncertainty(x)
        assert result is not None
        assert len(result) == 2

    def test_mc_dropout_estimate_from_estimate_uncertainty(self):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            pytest.skip("PyTorch no está instalado")

        from app.core_ml.uncertainty.mc_dropout import MCDropoutEstimator

        device = torch.device("cpu")
        model = nn.Sequential(
            nn.Conv2d(3, 2, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
        )
        model.train()
        estimator = MCDropoutEstimator(model, device=device, mc_samples=5)
        input_data = np.random.randn(2, 3, 8, 8).astype(np.float32)
        result = estimator.estimate_uncertainty(input_data)
        assert result is not None


class TestNoisyInferenceEstimator:
    def test_noise_inference_estimate_shape(self):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            pytest.skip("PyTorch no está instalado")

        from app.core_ml.uncertainty.noise_inference import NoisyInferenceEstimator

        device = torch.device("cpu")
        model = nn.Sequential(
            nn.Conv2d(3, 2, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
        )
        estimator = NoisyInferenceEstimator(model, device=device, n_samples=5, noise_std=0.01)
        x = torch.randn(2, 3, 8, 8)
        result = estimator.compute_uncertainty(x)
        assert result is not None
        assert len(result) == 2


class TestBaseUncertainty:
    def test_estimate_uncertainty_interface(self):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            pytest.skip("PyTorch no está instalado")

        from app.core_ml.uncertainty.base import BaseUncertaintyEstimator

        device = torch.device("cpu")
        model = nn.Sequential(
            nn.Conv2d(3, 2, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
        )

        class MinImpl(BaseUncertaintyEstimator):
            def compute_uncertainty(self, x, **kwargs):
                avg_probs = torch.softmax(self.model(x), dim=1)
                entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-8), dim=1)
                return avg_probs, entropy

        estimator = MinImpl(model, device)
        input_data = np.random.randn(2, 3, 8, 8).astype(np.float32)
        result = estimator.estimate_uncertainty(input_data)
        assert result is not None


class TestSklearnUncertainty:
    def test_entropy_sklearn(self):
        try:
            from app.core_ml.uncertainty.sklearn_uncertainty import entropy_uncertainty
        except ImportError:
            pytest.skip("sklearn_uncertainty no disponible")

        probs = np.ones((10, 5)) / 5
        result = entropy_uncertainty(probs)
        assert result.shape == (10,)
        assert np.all(result >= 0)

    def test_tree_variance(self):
        try:
            from sklearn.ensemble import RandomForestClassifier
            from app.core_ml.uncertainty.sklearn_uncertainty import tree_variance_uncertainty
        except (ImportError, AttributeError):
            pytest.skip("tree_variance_uncertainty no disponible")

        X = np.random.randn(20, 4)
        y = np.random.randint(0, 2, 20)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        result = tree_variance_uncertainty(model, X[:5])
        assert result.shape == (5,)
