"""
Tests for app.core_ml.preprocessing: save_pipeline with MLflow and load_pipeline fallback.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest


class TestSavePipeline:
    def test_save_pipeline_with_mlflow(self, monkeypatch):
        from app.core_ml.preprocessing import save_pipeline, build_pipeline

        config = {
            "steps": [
                {"type": "impute", "columns": ["age"], "strategy": "median"},
                {"type": "scale", "columns": ["age"], "strategy": "standard"},
            ]
        }
        pipeline = build_pipeline(config, ["age", "target"])

        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_123"
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_run

        monkeypatch.setattr("mlflow.start_run", lambda **kw: mock_context)
        monkeypatch.setattr("mlflow.sklearn.log_model", lambda model, artifact_path: None)
        monkeypatch.setattr("mlflow.set_tag", lambda key, value: None)
        monkeypatch.setattr("mlflow.set_experiment", lambda name: None)
        monkeypatch.setattr("mlflow.set_tracking_uri", lambda uri: None)

        mock_mlflow_svc = MagicMock()
        mock_mlflow_svc.tracking_uri = "mock://tracking"
        monkeypatch.setattr(
            "app.core_ml.preprocessing.MLFlowService",
            lambda: mock_mlflow_svc,
        )

        uri = save_pipeline(
            pipeline,
            "test_pipeline",
            "tenant_1",
            config={"target_column": "target", "steps": config["steps"]},
        )
        assert uri == "runs:/test_run_123/pipeline"

    def test_save_pipeline_without_config(self, monkeypatch):
        from app.core_ml.preprocessing import save_pipeline, build_pipeline

        config = {"steps": [{"type": "impute", "columns": ["age"], "strategy": "median"}]}
        pipeline = build_pipeline(config, ["age", "target"])

        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_456"
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_run

        monkeypatch.setattr("mlflow.start_run", lambda **kw: mock_context)
        monkeypatch.setattr("mlflow.sklearn.log_model", lambda model, artifact_path: None)
        monkeypatch.setattr("mlflow.set_tag", lambda key, value: None)
        monkeypatch.setattr("mlflow.set_experiment", lambda name: None)
        monkeypatch.setattr("mlflow.set_tracking_uri", lambda uri: None)

        mock_mlflow_svc = MagicMock()
        mock_mlflow_svc.tracking_uri = "mock://tracking"
        monkeypatch.setattr(
            "app.core_ml.preprocessing.MLFlowService",
            lambda: mock_mlflow_svc,
        )

        uri = save_pipeline(pipeline, "no_config_pipeline", "tenant_2")
        assert uri == "runs:/test_run_456/pipeline"


class TestLoadPipeline:
    def test_load_pipeline_local_joblib(self):
        import joblib
        from app.core_ml.preprocessing import build_pipeline, load_pipeline

        config = {
            "steps": [
                {"type": "impute", "columns": ["age"], "strategy": "median"},
                {"type": "scale", "columns": ["age"], "strategy": "standard"},
            ]
        }
        pipeline = build_pipeline(config, ["age", "target"])

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            joblib.dump(pipeline, tmp_path)
            loaded = load_pipeline(tmp_path)
            assert loaded is not None
            assert hasattr(loaded, "transform")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_load_pipeline_nonexistent_raises(self):
        from app.core_ml.preprocessing import load_pipeline

        with pytest.raises(FileNotFoundError):
            load_pipeline("/nonexistent/pipeline.joblib")

    def test_load_pipeline_mlflow_uri(self, monkeypatch):
        from app.core_ml.preprocessing import load_pipeline

        mock_pipeline = MagicMock()
        monkeypatch.setattr("mlflow.sklearn.load_model", lambda uri: mock_pipeline)

        mock_mlflow_svc = MagicMock()
        mock_mlflow_svc.tracking_uri = "mock://tracking"
        monkeypatch.setattr(
            "app.core_ml.preprocessing.MLFlowService",
            lambda: mock_mlflow_svc,
        )
        monkeypatch.setattr("mlflow.set_tracking_uri", lambda uri: None)

        loaded = load_pipeline("runs:/some_run/pipeline")
        assert loaded is mock_pipeline
