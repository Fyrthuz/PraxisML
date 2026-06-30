import numpy as np
import pandas as pd
import pytest


class TestSklearnTrainer:
    def test_sklearn_holdout_classification(self):
        from app.services.training_service import SklearnTrainer

        df = pd.DataFrame({
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "feature2": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            "target": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        })

        trainer = SklearnTrainer(tenant_id="test-tenant")
        result = trainer.train(
            df=df,
            target_column="target",
            algorithm="random_forest",
            task_type="classification",
            hyperparams={"n_estimators": 10, "max_depth": 3},
            validation_config={"strategy": "holdout", "test_size": 0.3},
            model_name="test-model",
        )

        assert "model" in result
        assert "metrics" in result
        assert "mlflow_run_id" in result
        assert result["metrics"]["accuracy"] >= 0.0
        assert result["metrics"]["accuracy"] <= 1.0

    def test_sklearn_holdout_regression(self):
        from app.services.training_service import SklearnTrainer

        np.random.seed(42)
        df = pd.DataFrame({
            "feature1": np.random.randn(50),
            "feature2": np.random.randn(50),
            "target": np.random.randn(50),
        })

        trainer = SklearnTrainer(tenant_id="test-tenant")
        result = trainer.train(
            df=df,
            target_column="target",
            algorithm="random_forest",
            task_type="regression",
            hyperparams={"n_estimators": 10},
            validation_config={"strategy": "holdout", "test_size": 0.3},
            model_name="test-reg-model",
        )

        assert "model" in result
        assert "metrics" in result
        assert "mse" in result["metrics"]

    def test_sklearn_cross_validation(self):
        from app.services.training_service import SklearnTrainer

        df = pd.DataFrame({
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] * 3,
            "feature2": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0] * 3,
            "target": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1] * 3,
        })

        trainer = SklearnTrainer(tenant_id="test-tenant")
        result = trainer.train(
            df=df,
            target_column="target",
            algorithm="logistic_regression",
            task_type="classification",
            hyperparams={"C": 1.0, "max_iter": 500},
            validation_config={"strategy": "cross_validation", "n_folds": 3},
            model_name="test-cv-model",
        )

        assert "model" in result
        assert "metrics" in result
        assert "cv_mean_accuracy" in result["metrics"]

    def test_sklearn_unknown_algorithm_raises(self):
        from app.services.training_service import SklearnTrainer

        trainer = SklearnTrainer(tenant_id="test-tenant")
        df = pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})

        with pytest.raises(ValueError):
            trainer.train(
                df=df, target_column="y", algorithm="nonexistent_algo", task_type="classification"
            )


class TestPyTorchTrainer:
    def test_pytorch_holdout_classification(self):
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch no está instalado")

        from app.services.training_service import PyTorchTrainer

        df = pd.DataFrame({
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] * 3,
            "feature2": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0] * 3,
            "target": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1] * 3,
        })

        trainer = PyTorchTrainer(tenant_id="test-tenant")
        result = trainer.train(
            df=df,
            target_column="target",
            algorithm="mlp",
            task_type="classification",
            hyperparams={"hidden_layers": [4, 2], "epochs": 5, "lr": 0.01},
            validation_config={"strategy": "holdout", "test_size": 0.3},
            model_name="test-pytorch-model",
        )

        assert "model" in result
        assert "metrics" in result
        assert "mlflow_run_id" in result
        assert result["metrics"]["accuracy"] >= 0.0
