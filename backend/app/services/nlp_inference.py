"""
Servicio de inferencia NLP (clasificación de texto, NER).

Carga el modelo desde ONNX (inferencia CPU rápida con onnxruntime) o,
como fallback, desde un checkpoint transformers guardado localmente.
Integra ModelCache para evitar recargas.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from app.core.config import settings
from app.services.model_cache import get_model_cache

logger = logging.getLogger(__name__)

_DEFAULT_MAX_SEQ_LENGTH = 128


def _resolve_artifact_path(path_str: Optional[str]) -> Optional[str]:
    """Resuelve rutas locales de artefactos NLP entre host y contenedor Docker."""
    if not path_str:
        return None
    if os.path.exists(path_str):
        return path_str
    marker = "/tenants/"
    if marker in path_str:
        tail = path_str.split(marker, 1)[1]
        candidate = Path(settings.DATA_DIR) / "tenants" / tail
        if candidate.exists():
            return str(candidate)
    rel_candidate = Path(settings.DATA_DIR) / path_str.lstrip("/")
    if rel_candidate.exists():
        return str(rel_candidate)
    return path_str


class NLPInferenceService:
    """Inferencia unificada para modelos NLP de la plataforma."""

    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache
        self._cache = get_model_cache() if use_cache else None

    # ──────────────────────────────────────────────────────────────────────
    # Carga de modelos
    # ──────────────────────────────────────────────────────────────────────

    def load_model(self, ml_model, device: Optional[str] = None) -> Dict[str, Any]:
        """
        Carga el modelo NLP.

        Devuelve un handle dict con:
          kind: "onnx" | "transformers"
          session / model, tokenizer, task_type, label_map, id2label, max_seq_length
        """
        metadata = ml_model.metrics_metadata or {}
        onnx_path = _resolve_artifact_path(metadata.get("onnx_path"))
        model_path = _resolve_artifact_path(metadata.get("model_path"))
        tokenizer_path = _resolve_artifact_path(metadata.get("tokenizer_path")) or model_path
        task_type = metadata.get("task_type", "text_classification")
        label_map = metadata.get("label_map") or {}
        id2label = metadata.get("id2label") or {str(v): k for k, v in label_map.items()}
        max_seq_length = int(metadata.get("max_seq_length", _DEFAULT_MAX_SEQ_LENGTH))

        cache_key = f"nlp:{ml_model.id}:{onnx_path or model_path or ml_model.mlflow_run_id}"
        if self._cache:
            handle = self._cache.get(cache_key)
            if handle is not None:
                logger.info("Modelo NLP %s cargado desde cache", ml_model.id)
                return handle

        handle = None
        if onnx_path:
            try:
                handle = self._load_onnx(onnx_path, tokenizer_path, task_type, label_map, id2label, max_seq_length)
            except Exception as exc:
                logger.warning("Carga ONNX falló (%s), intentando transformers", exc)

        if handle is None:
            handle = self._load_transformers(
                model_path, tokenizer_path, task_type, label_map, id2label,
                max_seq_length, device,
            )

        if self._cache:
            self._cache.set(cache_key, handle, ttl=600)
        return handle

    def _load_onnx(
        self, onnx_path: str, tokenizer_path: str, task_type: str,
        label_map: Dict, id2label: Dict, max_seq_length: int,
    ) -> Dict[str, Any]:
        import onnxruntime
        from transformers import AutoTokenizer

        session = onnxruntime.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        return {
            "kind": "onnx",
            "session": session,
            "tokenizer": tokenizer,
            "task_type": task_type,
            "label_map": label_map,
            "id2label": id2label,
            "max_seq_length": max_seq_length,
        }

    def _load_transformers(
        self, model_path: str, tokenizer_path: str, task_type: str,
        label_map: Dict, id2label: Dict, max_seq_length: int, device: Optional[str],
    ) -> Dict[str, Any]:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoModelForTokenClassification,
            AutoTokenizer,
        )

        if task_type == "token_classification":
            model_cls = AutoModelForTokenClassification
        else:
            model_cls = AutoModelForSequenceClassification

        model = model_cls.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model.to(dev)
        model.eval()
        return {
            "kind": "transformers",
            "model": model,
            "tokenizer": tokenizer,
            "task_type": task_type,
            "label_map": label_map,
            "id2label": id2label,
            "max_seq_length": max_seq_length,
            "device": dev,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Inferencia
    # ──────────────────────────────────────────────────────────────────────

    def predict(self, handle: Dict[str, Any], text: str) -> Dict[str, Any]:
        """Predice sobre un texto libre. Devuelve {label, confidence, entities, tokens, latency_ms}."""
        t0 = time.perf_counter()
        logits, enc = self._forward(handle, text)
        latency_ms = (time.perf_counter() - t0) * 1000

        if handle["task_type"] == "token_classification":
            return self._postprocess_ner(handle, logits, enc, latency_ms)
        return self._postprocess_classification(handle, logits, enc, latency_ms)

    def predict_batch(
        self, handle: Dict[str, Any], df, text_column: str
    ) -> List[Dict[str, Any]]:
        """Predice sobre todas las filas de un DataFrame."""
        results = []
        for text in df[text_column].astype(str).tolist():
            try:
                results.append(self.predict(handle, text))
            except Exception as exc:
                results.append({"error": str(exc), "text": text})
        return results

    def _forward(self, handle: Dict[str, Any], text: str):
        tokenizer = handle["tokenizer"]
        max_seq_length = handle["max_seq_length"]

        if handle["kind"] == "onnx":
            enc = tokenizer(
                text, truncation=True, padding=True,
                max_length=max_seq_length, return_tensors="np",
            )
            feed = {k: np.asarray(v) for k, v in enc.items()}
            logits = handle["session"].run(None, feed)[0]
            return logits, enc

        import torch

        enc = tokenizer(
            text, truncation=True, padding=True,
            max_length=max_seq_length, return_tensors="pt",
        )
        enc = {k: v.to(handle["device"]) for k, v in enc.items()}
        with torch.no_grad():
            out = handle["model"](**enc)
        logits = out.logits.cpu().numpy()
        return logits, enc

    def _postprocess_classification(
        self, handle, logits, enc, latency_ms: float
    ) -> Dict[str, Any]:
        probs = self._softmax(logits[0])
        idx = int(np.argmax(probs))
        id2label = handle["id2label"] or {}
        label = id2label.get(idx) or id2label.get(str(idx)) or str(idx)
        tokens = self._get_tokens(handle, enc)

        return {
            "label": label,
            "confidence": float(probs[idx]),
            "entities": [],
            "tokens": tokens,
            "latency_ms": round(latency_ms, 2),
        }

    def _postprocess_ner(
        self, handle, logits, enc, latency_ms: float
    ) -> Dict[str, Any]:
        id2label = handle["id2label"] or {}
        tokens = self._get_tokens(handle, enc)
        pred_ids = np.argmax(logits[0], axis=-1)
        labels = [id2label.get(int(p), str(p)) for p in pred_ids]
        entities = self._build_entities(tokens, labels)

        return {
            "label": None,
            "confidence": None,
            "entities": entities,
            "tokens": tokens,
            "latency_ms": round(latency_ms, 2),
        }

    def _get_tokens(self, handle, enc) -> List[str]:
        tokenizer = handle["tokenizer"]
        try:
            ids = enc["input_ids"]
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            return tokenizer.convert_ids_to_tokens(ids[0])
        except Exception:
            try:
                return tokenizer.tokenize(enc.get("text", ""))
            except Exception:
                return []

    @staticmethod
    def _build_entities(tokens: List[str], labels: List[str]) -> List[Dict[str, Any]]:
        """Agrupa tokens BIO en entidades [{entity, label, score?}]."""
        entities = []
        current: Optional[Dict[str, Any]] = None
        for token, label in zip(tokens, labels):
            if token in ("[CLS]", "[SEP]", "[PAD]") or token.startswith("##"):
                continue
            if label.startswith("B-"):
                if current:
                    entities.append(current)
                current = {"entity": token, "label": label[2:]}
            elif label.startswith("I-") and current:
                current["entity"] += f" {token}"
            elif label == "O" and current:
                entities.append(current)
                current = None
        if current:
            entities.append(current)
        return entities

    @staticmethod
    def _softmax(scores: np.ndarray) -> np.ndarray:
        exp = np.exp(scores - np.max(scores))
        return exp / exp.sum()

    # ──────────────────────────────────────────────────────────────────────
    # Observabilidad
    # ──────────────────────────────────────────────────────────────────────

    def record_usage(
        self, db, tenant_id: str, model_id: str, tokens_in: int,
        latency_ms: float, tags: Optional[dict] = None,
    ) -> None:
        """Registra uso y coste de una inferencia NLP (dominio nlp)."""
        from datetime import datetime, timezone

        from app.observability.cost_calculator import get_cost_calculator
        from app.observability.metrics import (
            nlp_inference_latency_ms,
            nlp_predictions_total,
        )
        from app.observability.usage_collector import get_usage_collector

        tags = tags or {}
        get_usage_collector().record_inference(
            db=db,
            tenant_id=tenant_id,
            model_id=model_id,
            domain="nlp",
            latency_ms=latency_ms,
            tokens_input=tokens_in,
            tokens_output=0,
            tags=tags,
        )
        nlp_inference_latency_ms.labels(
            tenant_id=tenant_id,
            model_id=model_id,
            task_type=tags.get("task_type", "unknown"),
        ).observe(latency_ms)
        nlp_predictions_total.labels(
            tenant_id=tenant_id,
            model_id=model_id,
            task_type=tags.get("task_type", "unknown"),
            status="success",
        ).inc()

        try:
            calculator = get_cost_calculator()
            cost = calculator.estimate_inference_cost(
                db=db, domain="nlp", latency_ms=latency_ms, gpu_type=None
            )
            if cost > 0:
                now = datetime.now(timezone.utc)
                calculator.record_cost(
                    db=db, tenant_id=tenant_id, cost_type="inference", amount=cost,
                    source=f"nlp:{model_id}", period_start=now, period_end=now,
                    model_id=model_id, metadata=tags,
                )
        except Exception as exc:
            logger.warning("Cost recording failed (non-critical): %s", exc)


_inference_service_instance: Optional[NLPInferenceService] = None


def get_nlp_inference_service(use_cache: bool = True) -> NLPInferenceService:
    """Factory singleton del NLPInferenceService."""
    global _inference_service_instance
    if _inference_service_instance is None:
        _inference_service_instance = NLPInferenceService(use_cache=use_cache)
    return _inference_service_instance
