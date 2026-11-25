"""
Tests unitarios para organización jerárquica (MLflow y MinIO), resolución de rutas
NLP entre host/Docker y guardias de compatibilidad entre dominios Tabular y NLP.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from app.api.routes.v1.nlp.inference import _get_nlp_model
from app.api.routes.v1.tabular.predictions import _get_tabular_model
from app.services.mlflow_service import MLFlowService
from app.services.nlp_inference import _resolve_artifact_path
from app.services.storage_service import StorageKeyBuilder


class TestStorageKeyBuilder:
    def test_hierarchical_dataset_keys(self):
        tid = "185e7404-04a9-492c-86a3-bb895df6b322"
        assert (
            StorageKeyBuilder.dataset_raw(tid, "titanic.csv")
            == f"tenants/{tid}/datasets/raw/titanic.csv"
        )
        assert (
            StorageKeyBuilder.dataset_processed(tid, "titanic_v2.csv")
            == f"tenants/{tid}/datasets/processed/titanic_v2.csv"
        )
        assert (
            StorageKeyBuilder.dataset_pipeline(tid, "feat_v1.csv")
            == f"tenants/{tid}/datasets/pipelines/feat_v1.csv"
        )
        assert (
            StorageKeyBuilder.dataset_config(tid, "my data", "cfg.json")
            == f"tenants/{tid}/datasets/configs/my_data_config_cfg.json"
        )
        assert (
            StorageKeyBuilder.model_artifact(tid, "m1", "model.onnx")
            == f"tenants/{tid}/artifacts/models/m1/model.onnx"
        )
        assert (
            StorageKeyBuilder.prediction_output(tid, "p1", "out.csv")
            == f"tenants/{tid}/predictions/p1/out.csv"
        )

    def test_sanitizes_path_traversal(self):
        key = StorageKeyBuilder.dataset_raw("t1", "../../etc/passwd")
        assert key == "tenants/t1/datasets/raw/passwd"


class TestMLFlowExperimentNaming:
    def test_canonical_domains_with_uuid_fallback(self):
        tid = "185e7404-04a9-492c-86a3-bb895df6b322"
        assert MLFlowService.get_experiment_name(tid, "training") == "tenant-185e7404/tabular-training"
        assert MLFlowService.get_experiment_name(tid, "nlp_training") == "tenant-185e7404/nlp-training"
        assert MLFlowService.get_experiment_name(tid, "preprocessing") == "tenant-185e7404/preprocessing"
        assert MLFlowService.get_experiment_name(tid, "pipelines") == "tenant-185e7404/pipelines"

    def test_uses_tenant_slug_when_provided(self):
        assert (
            MLFlowService.get_experiment_name("uuid-1", "pipelines", tenant_slug="Acme Corp")
            == "acme-corp/pipelines"
        )


class TestNLPArtifactPathResolution:
    def test_remaps_host_path_to_container_data_dir(self, tmp_path, monkeypatch):
        from app.core.config import settings

        target = tmp_path / "tenants" / "t1" / "nlp_models" / "onnx_abc" / "model.onnx"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"onnx")
        monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path))

        host_path = "/home/dev/PraxisML/backend/data/tenants/t1/nlp_models/onnx_abc/model.onnx"
        resolved = _resolve_artifact_path(host_path)
        assert resolved == str(target)
        assert Path(resolved).exists()


class TestCrossDomainModelGuards:
    @staticmethod
    def _mock_db(model):
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = model
        return db

    def test_tabular_guard_rejects_nlp_model(self):
        nlp_model = SimpleNamespace(
            id="m-nlp",
            tenant_id="t1",
            metrics_metadata={"domain": "nlp", "framework": "huggingface", "task_type": "sentiment_analysis"},
        )
        with pytest.raises(HTTPException) as exc:
            _get_tabular_model(self._mock_db(nlp_model), "m-nlp", "t1")
        assert exc.value.status_code == 400

    def test_tabular_guard_accepts_tabular_model(self):
        tab_model = SimpleNamespace(
            id="m-tab",
            tenant_id="t1",
            metrics_metadata={"task_type": "classification", "algorithm": "random_forest"},
        )
        assert _get_tabular_model(self._mock_db(tab_model), "m-tab", "t1") is tab_model

    def test_nlp_guard_rejects_tabular_model_even_when_domain_is_none(self):
        tab_model = SimpleNamespace(
            id="m-tab",
            tenant_id="t1",
            metrics_metadata={"task_type": "classification", "algorithm": "random_forest"},
        )
        with pytest.raises(HTTPException) as exc:
            _get_nlp_model(self._mock_db(tab_model), "m-tab", "t1")
        assert exc.value.status_code == 400

    def test_nlp_guard_accepts_nlp_model(self):
        nlp_model = SimpleNamespace(
            id="m-nlp",
            tenant_id="t1",
            metrics_metadata={"domain": "nlp", "framework": "huggingface", "task_type": "text_classification"},
        )
        assert _get_nlp_model(self._mock_db(nlp_model), "m-nlp", "t1") is nlp_model
