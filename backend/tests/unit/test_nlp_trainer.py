"""
Tests unitarios del NLPTrainer (mocks: transformers, mlflow, onnx).
Sin descargas de modelos ni conexión a MLflow.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.services.nlp_trainer import NLPTrainer


class MockRun:
    info = SimpleNamespace(run_id="run123456")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class FakeTokenizer:
    def __call__(self, texts, **kwargs):
        n = len(texts) if isinstance(texts, list) else 1
        return {"input_ids": np.zeros((n, 8), dtype=int), "attention_mask": np.ones((n, 8), dtype=int)}

    def word_ids(self, batch_index=0):
        return [0, 0, 1, 2, None, None, None, None]

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)


class FakeModel:
    def eval(self):
        return self

    def to(self, device):
        return self


class FakeTrainer:
    def __init__(self, *args, **kwargs):
        pass

    def train(self):
        return None

    def evaluate(self):
        return {
            "eval_loss": 0.42,
            "eval_accuracy": 0.85,
            "eval_precision": 0.80,
            "eval_recall": 0.78,
            "eval_f1": 0.79,
            "eval_runtime": 1.2,
        }

    def save_model(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)


def _sample_df():
    texts = [
        "Great product, I love it",
        "Terrible experience, would not buy",
        "Excellent quality and fast shipping",
        "Waste of money, very disappointed",
        "Amazing, works perfectly",
        "Bad customer service, avoid",
    ] * 3
    labels = ["pos", "neg", "pos", "neg", "pos", "neg"] * 3
    return pd.DataFrame({"text": texts, "label": labels})


@pytest.fixture
def mock_transformers(monkeypatch):
    """Parchea las clases transformers en el espacio de nombres de nlp_trainer."""
    from sklearn import model_selection as ms

    monkeypatch.setattr(ms, "train_test_split", lambda *a, **k: (a[0].iloc[:10], a[0].iloc[10:]))
    monkeypatch.setattr(
        "app.services.nlp_trainer.AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda *a, **k: FakeTokenizer()),
    )
    monkeypatch.setattr(
        "app.services.nlp_trainer.AutoModelForSequenceClassification",
        SimpleNamespace(from_pretrained=lambda *a, **k: FakeModel()),
    )
    monkeypatch.setattr(
        "app.services.nlp_trainer.AutoModelForTokenClassification",
        SimpleNamespace(from_pretrained=lambda *a, **k: FakeModel()),
    )
    monkeypatch.setattr("app.services.nlp_trainer.Trainer", FakeTrainer)
    monkeypatch.setattr(
        "app.services.nlp_trainer.TrainingArguments", lambda **k: SimpleNamespace(**k)
    )
    return True


@pytest.fixture
def mock_mlflow(monkeypatch):
    import mlflow

    monkeypatch.setattr(mlflow, "start_run", lambda **k: MockRun())
    monkeypatch.setattr(mlflow, "set_experiment", lambda *a, **k: None)
    monkeypatch.setattr(mlflow, "set_tags", lambda *a, **k: None)
    monkeypatch.setattr(mlflow, "log_params", lambda *a, **k: None)
    monkeypatch.setattr(mlflow, "log_metric", lambda *a, **k: None)
    monkeypatch.setattr(mlflow, "log_artifact", lambda *a, **k: None)
    monkeypatch.setattr(mlflow, "log_artifacts", lambda *a, **k: None)
    mlflow.transformers.autolog = lambda **k: None
    return mlflow


@pytest.fixture
def mock_onnx(monkeypatch):
    monkeypatch.setattr(
        "app.services.nlp_trainer.export_model_to_onnx",
        lambda model, tokenizer, out_dir, max_seq_length=128: Path(out_dir) / "model.onnx",
    )
    return


def test_train_holdout_classification(mock_transformers, mock_mlflow, mock_onnx, tmp_path):
    trainer = NLPTrainer(tenant_id="t1")
    result = trainer.train(
        df=_sample_df(),
        text_column="text",
        label_column="label",
        algorithm="distilbert",
        task_type="text_classification",
        hyperparams={"epochs": 1},
    )
    assert result["mlflow_run_id"] == "run123456"
    assert result["validation_strategy"] == "holdout"
    assert result["label_map"] == {"neg": 0, "pos": 1}
    assert result["feature_names"] == ["text"]
    assert result["metrics"]["eval_accuracy"] == 0.85
    assert Path(result["model_path"]).exists()
    assert result["onnx_path"].endswith("model.onnx")


def test_train_holdout_sentiment(mock_transformers, mock_mlflow, mock_onnx):
    trainer = NLPTrainer(tenant_id="t1")
    result = trainer.train(
        df=_sample_df(),
        text_column="text",
        label_column="label",
        algorithm="distilbert",
        task_type="sentiment_analysis",
    )
    assert result["label_map"] == {"neg": 0, "pos": 1}


def test_train_cv(mock_transformers, mock_mlflow, mock_onnx):
    from sklearn import model_selection as ms

    ms.StratifiedKFold = lambda n_splits, shuffle, random_state: SimpleNamespace(split=lambda *a, **k: [(list(range(0, 9)), list(range(9, 18))), (list(range(9, 18)), list(range(0, 9)))])
    trainer = NLPTrainer(tenant_id="t1")
    result = trainer.train(
        df=_sample_df(),
        text_column="text",
        label_column="label",
        algorithm="distilbert",
        task_type="text_classification",
        validation_config={"strategy": "cross_validation", "n_folds": 2},
    )
    assert result["validation_strategy"] == "cross_validation"
    assert result["n_folds"] == 2
    assert "cv_mean_eval_accuracy" in result["metrics"]


def test_train_missing_column_raises(mock_transformers, mock_mlflow, mock_onnx):
    trainer = NLPTrainer(tenant_id="t1")
    with pytest.raises(ValueError):
        trainer.train(
            df=_sample_df(),
            text_column="does_not_exist",
            label_column="label",
        )


def test_train_ner_requires_labels(mock_transformers, mock_mlflow, mock_onnx):
    trainer = NLPTrainer(tenant_id="t1")
    with pytest.raises(ValueError):
        trainer.train(
            df=_sample_df(),
            text_column="text",
            task_type="token_classification",
        )
