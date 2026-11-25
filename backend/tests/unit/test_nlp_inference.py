"""
Tests unitarios del NLPInferenceService (post-procesado y pipeline con mocks).
"""

import numpy as np

from app.domains.nlp.inference import NLPInferenceService


def _handle(task_type="text_classification"):
    return {
        "kind": "onnx",
        "session": None,
        "tokenizer": None,
        "task_type": task_type,
        "label_map": {"neg": 0, "pos": 1},
        "id2label": {0: "neg", 1: "pos"},
        "max_seq_length": 128,
    }


def test_classification_postprocess():
    svc = NLPInferenceService(use_cache=False)
    handle = _handle()
    logits = np.array([[0.1, 1.5]], dtype=np.float32)  # 2 clases
    enc = {"input_ids": [[101, 2023, 102]]}
    result = svc._postprocess_classification(handle, logits, enc, 12.5)
    assert result["label"] == "pos"
    assert 0.5 < result["confidence"] <= 1.0
    assert result["latency_ms"] == 12.5
    assert "tokens" in result


def test_ner_postprocess_builds_entities():
    svc = NLPInferenceService(use_cache=False)
    handle = {
        "kind": "onnx",
        "session": None,
        "tokenizer": None,
        "task_type": "token_classification",
        "label_map": {"O": 0, "B-PER": 1, "I-PER": 2},
        "id2label": {0: "O", 1: "B-PER", 2: "I-PER"},
        "max_seq_length": 128,
    }
    # 4 tokens: [CLS] Juan Perez [SEP]
    logits = np.array([[
        [0.9, 0.05, 0.05],  # O
        [0.1, 0.8, 0.1],    # B-PER
        [0.1, 0.2, 0.7],    # I-PER
        [0.9, 0.05, 0.05],  # O
    ]], dtype=np.float32)
    enc = {"input_ids": [[101, 1234, 5678, 102]]}
    # mock de _get_tokens
    svc._get_tokens = lambda *a: ["[CLS]", "Juan", "Perez", "[SEP]"]
    result = svc._postprocess_ner(handle, logits, enc, 5.0)
    assert result["label"] is None
    assert len(result["entities"]) == 1
    assert result["entities"][0]["label"] == "PER"
    assert "Juan" in result["entities"][0]["entity"]


def test_predict_calls_forward(monkeypatch):
    svc = NLPInferenceService(use_cache=False)
    handle = _handle()
    monkeypatch.setattr(
        svc, "_forward", lambda h, t: (np.array([[0.1, 2.0]], dtype=np.float32), {"input_ids": [[101, 102]]})
    )
    monkeypatch.setattr(svc, "_get_tokens", lambda *a: ["[CLS]", "word", "[SEP]"])
    result = svc.predict(handle, "great product")
    assert result["label"] == "pos"
    assert result["latency_ms"] >= 0


def test_predict_batch(monkeypatch):
    import pandas as pd

    svc = NLPInferenceService(use_cache=False)
    handle = _handle()
    monkeypatch.setattr(
        svc, "_forward", lambda h, t: (np.array([[0.1, 2.0]], dtype=np.float32), {"input_ids": [[101, 102]]})
    )
    monkeypatch.setattr(svc, "_get_tokens", lambda *a: ["[CLS]", "word", "[SEP]"])
    df = pd.DataFrame({"text": ["a", "b"]})
    results = svc.predict_batch(handle, df, "text")
    assert len(results) == 2
    assert all(r["label"] == "pos" for r in results)


def test_load_model_onnx_path(monkeypatch, tmp_path):
    from types import SimpleNamespace

    ml_model = SimpleNamespace(
        id="m1",
        mlflow_run_id="run1",
        metrics_metadata={
            "domain": "nlp",
            "onnx_path": str(tmp_path / "model.onnx"),
            "tokenizer_path": str(tmp_path / "tokenizer"),
            "task_type": "text_classification",
            "label_map": {"a": 0, "b": 1},
            "max_seq_length": 64,
        },
    )
    import onnxruntime
    import transformers

    monkeypatch.setattr(onnxruntime, "InferenceSession", lambda *a, **k: SimpleNamespace(run=lambda *a, **k: [np.zeros((1, 2))]))
    monkeypatch.setattr(
        transformers, "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda *a, **k: SimpleNamespace(
            __call__=lambda *a, **k: {"input_ids": [[0, 1]]},
            convert_ids_to_tokens=lambda ids: ["a", "b"],
            tokenize=lambda t: ["a", "b"],
        )),
    )

    svc = NLPInferenceService(use_cache=False)
    handle = svc.load_model(ml_model)
    assert handle["kind"] == "onnx"
    assert handle["task_type"] == "text_classification"
