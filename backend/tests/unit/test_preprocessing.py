import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_column_names():
    return ["age", "income", "city", "target"]


class TestBuildPipeline:
    def test_build_basic_pipeline(self, sample_column_names):
        from app.core_ml.preprocessing import build_pipeline

        config = {
            "steps": [
                {"type": "impute", "columns": ["age"], "strategy": "median"},
                {"type": "scale", "columns": ["age"], "strategy": "standard"},
            ]
        }
        pipeline = build_pipeline(config, sample_column_names)
        assert pipeline is not None
        assert hasattr(pipeline, "transform")

    def test_build_with_all_step_types(self, sample_column_names):
        from app.core_ml.preprocessing import build_pipeline

        config = {
            "steps": [
                {"type": "impute", "columns": ["age", "income"], "strategy": "mean"},
                {"type": "scale", "columns": ["age", "income"], "strategy": "minmax"},
                {"type": "encode", "columns": ["city"], "strategy": "onehot"},
            ]
        }
        pipeline = build_pipeline(config, sample_column_names)
        assert pipeline is not None

    def test_build_with_drop_step(self):
        from app.core_ml.preprocessing import build_pipeline

        config = {
            "steps": [
                {"type": "drop", "columns": ["id"]},
                {"type": "impute", "columns": ["age"], "strategy": "median"},
            ]
        }
        pipeline = build_pipeline(config, ["id", "age", "target"])
        assert pipeline is not None

    def test_build_empty_steps_raises(self, sample_column_names):
        from app.core_ml.preprocessing import build_pipeline

        with pytest.raises(ValueError):
            build_pipeline({"steps": []}, sample_column_names)

    def test_build_unknown_step_type_raises(self, sample_column_names):
        from app.core_ml.preprocessing import build_pipeline

        with pytest.raises(ValueError):
            build_pipeline({"steps": [{"type": "unknown", "columns": ["x"], "strategy": "none"}]}, sample_column_names)


class TestApplyPipeline:
    def test_apply_pipeline_returns_X_y(self):
        from app.core_ml.preprocessing import apply_pipeline, build_pipeline

        df = pd.DataFrame({"age": [1, 2, 3], "target": [0, 1, 0]})
        config = {"steps": [{"type": "impute", "columns": ["age"], "strategy": "median"}]}
        pipeline = build_pipeline(config, ["age", "target"])
        X, y = apply_pipeline(pipeline, df, target_column="target")
        assert X.shape[1] == 1
        assert len(y) == 3

    def test_apply_pipeline_all_numeric_columns(self):
        from app.core_ml.preprocessing import apply_pipeline, build_pipeline

        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "target": [0, 1, 0]})
        config = {
            "steps": [
                {"type": "scale", "columns": ["a", "b"], "strategy": "standard"},
            ]
        }
        pipeline = build_pipeline(config, ["a", "b", "target"])
        X, y = apply_pipeline(pipeline, df, target_column="target")
        assert X.shape[1] == 2
        assert np.allclose(X.mean(axis=0), 0, atol=1e-7)

    def test_apply_pipeline_no_target_raises(self):
        from app.core_ml.preprocessing import apply_pipeline, build_pipeline

        df = pd.DataFrame({"age": [1, 2, 3]})
        config = {"steps": [{"type": "impute", "columns": ["age"], "strategy": "median"}]}
        pipeline = build_pipeline(config, ["age"])
        with pytest.raises(ValueError):
            apply_pipeline(pipeline, df, target_column="nonexistent")

    def test_apply_pipeline_with_imputation(self):
        from app.core_ml.preprocessing import apply_pipeline, build_pipeline

        df = pd.DataFrame({"age": [1, None, 3], "target": [0, 1, 0]})
        config = {"steps": [{"type": "impute", "columns": ["age"], "strategy": "median"}]}
        pipeline = build_pipeline(config, ["age", "target"])
        X, y = apply_pipeline(pipeline, df, target_column="target")
        assert not np.any(np.isnan(X))


class TestSaveLoadPipeline:
    def test_save_and_load_pipeline_local(self):
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

    def test_load_nonexistent_raises(self):
        from app.core_ml.preprocessing import load_pipeline

        with pytest.raises(FileNotFoundError):
            load_pipeline("/nonexistent/pipeline.joblib")
