"""
Servicio de embeddings semánticos (sentence-transformers) y búsqueda semántica.

Almacena embeddings en `model_embeddings` (pgvector en PostgreSQL, JSON en SQLite)
y ejecuta búsqueda por similitud coseno.
"""

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from app.core.config import settings
from app.models.embedding import ModelEmbedding
from app.services.model_cache import get_model_cache

logger = logging.getLogger(__name__)

_MODEL_KEY = "embedding_model"
_MODEL_TTL = 3600


class EmbeddingService:
    """Singleton que genera embeddings y gestiona el índice semántico."""

    def __init__(self, use_cache: bool = True):
        self._model = None
        self.use_cache = use_cache
        self._cache = get_model_cache() if use_cache else None

    # ──────────────────────────────────────────────────────────────────────
    # Embeddings
    # ──────────────────────────────────────────────────────────────────────

    def _get_model(self):
        if self._model is None:
            if self._cache:
                cached = self._cache.get(_MODEL_KEY)
                if cached is not None:
                    self._model = cached
                    return cached
            from sentence_transformers import SentenceTransformer

            logger.info("Cargando modelo de embeddings %s...", settings.EMBEDDING_MODEL_NAME)
            self._model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
            if self._cache:
                self._cache.set(_MODEL_KEY, self._model, ttl=_MODEL_TTL)
        return self._model

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embeddings normalizados (n, EMBEDDING_DIM)."""
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.zeros((0, settings.EMBEDDING_DIM), dtype=np.float32)
        model = self._get_model()
        vec = model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )
        return np.asarray(vec, dtype=np.float32)

    def embed_text(self, text: str) -> List[float]:
        return self.embed_texts([text])[0].tolist()

    # ──────────────────────────────────────────────────────────────────────
    # Construcción de metadata / texto indexado
    # ──────────────────────────────────────────────────────────────────────

    def _indexed_text(self, metadata: Dict[str, Any]) -> str:
        parts = [
            metadata.get("model_name")
            or metadata.get("name")
            or metadata.get("dataset_name", ""),
            metadata.get("description", ""),
            metadata.get("algorithm", ""),
            metadata.get("task_type", ""),
            metadata.get("stage", ""),
        ]
        return " ".join(p for p in parts if p) or "PraxisML"

    def _build_model_metadata(self, ml_model) -> Dict[str, Any]:
        meta = ml_model.metrics_metadata or {}
        metrics = meta.get("metrics", {}) or {}
        return {
            "index_type": "model",
            "tenant_id": ml_model.tenant_id,
            "model_id": ml_model.id,
            "model_name": ml_model.name,
            "description": ml_model.description or "",
            "domain": meta.get("domain", "tabular"),
            "algorithm": meta.get("algorithm", ""),
            "task_type": meta.get("task_type", ""),
            "framework": meta.get("framework", ""),
            "stage": ml_model.stage,
            "mlflow_run_id": ml_model.mlflow_run_id,
            "mlflow_registry_name": ml_model.mlflow_registry_name,
            "mlflow_version": ml_model.mlflow_version,
            "version": ml_model.version,
            "text_column": meta.get("text_column", ""),
            "label_column": meta.get("label_column", ""),
            "label_map": meta.get("label_map", {}),
            "metrics": metrics,
            "dataset_id": meta.get("dataset_id", ""),
            "dataset_name": meta.get("dataset_name", ""),
            "embedding_model": settings.EMBEDDING_MODEL_NAME,
            "text_preview": "",
        }

    def _build_dataset_metadata(
        self, dataset, column_types_analysis: Optional[dict] = None
    ) -> Dict[str, Any]:
        return {
            "index_type": "dataset",
            "tenant_id": dataset.tenant_id,
            "dataset_id": dataset.id,
            "name": dataset.name,
            "description": dataset.description or "",
            "file_type": dataset.file_type,
            "num_rows": dataset.num_rows,
            "num_columns": dataset.num_columns,
            "column_names": dataset.column_names or [],
            "domain": "tabular",
            "version": dataset.version,
            "column_types_analysis": column_types_analysis or {},
            "embedding_model": settings.EMBEDDING_MODEL_NAME,
            "text_preview": "",
        }

    # ──────────────────────────────────────────────────────────────────────
    # Indexado (upsert por entidad)
    # ──────────────────────────────────────────────────────────────────────

    def index_model(self, db, ml_model) -> None:
        metadata = self._build_model_metadata(ml_model)
        metadata["text_preview"] = self._indexed_text(metadata)[:200]
        vector = self.embed_texts([self._indexed_text(metadata)])[0].tolist()
        self._upsert(
            db,
            tenant_id=ml_model.tenant_id,
            model_id=ml_model.id,
            vector=vector,
            metadata=metadata,
        )
        self._record_op(db, ml_model.tenant_id, "index")

    def index_dataset(
        self, db, dataset, column_types_analysis: Optional[dict] = None
    ) -> None:
        metadata = self._build_dataset_metadata(dataset, column_types_analysis)
        metadata["text_preview"] = self._indexed_text(metadata)[:200]
        vector = self.embed_texts([self._indexed_text(metadata)])[0].tolist()
        self._upsert(
            db,
            tenant_id=dataset.tenant_id,
            dataset_id=dataset.id,
            vector=vector,
            metadata=metadata,
        )
        self._record_op(db, dataset.tenant_id, "index")

    def _upsert(
        self, db, tenant_id: str, vector: List[float], metadata: Dict[str, Any],
        model_id: Optional[str] = None, dataset_id: Optional[str] = None,
    ) -> None:
        existing = None
        if model_id:
            existing = (
                db.query(ModelEmbedding)
                .filter(
                    ModelEmbedding.tenant_id == tenant_id,
                    ModelEmbedding.model_id == model_id,
                )
                .first()
            )
        elif dataset_id:
            existing = (
                db.query(ModelEmbedding)
                .filter(
                    ModelEmbedding.tenant_id == tenant_id,
                    ModelEmbedding.dataset_id == dataset_id,
                )
                .first()
            )

        if existing:
            existing.embedding = vector
            existing.metadata_json = metadata
        else:
            db.add(
                ModelEmbedding(
                    tenant_id=tenant_id,
                    model_id=model_id,
                    dataset_id=dataset_id,
                    embedding=vector,
                    metadata_json=metadata,
                )
            )
        db.flush()

    def _record_op(self, db, tenant_id: str, operation: str) -> None:
        try:
            from app.observability.metrics import embeddings_total

            embeddings_total.labels(tenant_id=tenant_id, operation=operation).inc()
        except Exception as exc:
            logger.warning("Embeddings metric failed (non-critical): %s", exc)

    # ──────────────────────────────────────────────────────────────────────
    # Búsqueda semántica
    # ──────────────────────────────────────────────────────────────────────

    def search(
        self, db, tenant_id: str, query: str, top_k: int = 20,
        index: str = "models",
    ) -> List[Dict[str, Any]]:
        """Búsqueda semántica sobre modelos (default) o datasets del tenant."""
        t0 = time.perf_counter()
        qvec = self.embed_texts([query])[0].tolist()
        index_type = "model" if index == "models" else "dataset"
        dialect = db.bind.dialect.name if db.bind else "sqlite"

        if dialect == "postgresql":
            results = self._search_postgres(db, tenant_id, qvec, index_type, top_k)
        else:
            results = self._search_bruteforce(db, tenant_id, qvec, index_type, top_k)

        latency_ms = (time.perf_counter() - t0) * 1000
        try:
            from app.observability.metrics import (
                embeddings_total,
                semantic_search_latency_ms,
            )

            embeddings_total.labels(tenant_id=tenant_id, operation="search").inc()
            semantic_search_latency_ms.labels(
                tenant_id=tenant_id, index_type=index_type
            ).observe(latency_ms)
        except Exception as exc:
            logger.warning("Search metrics failed (non-critical): %s", exc)

        return results

    def _search_postgres(
        self, db, tenant_id: str, qvec: List[float], index_type: str, top_k: int,
    ) -> List[Dict[str, Any]]:
        rows = (
            db.query(ModelEmbedding)
            .filter(
                ModelEmbedding.tenant_id == tenant_id,
                ModelEmbedding.model_id.isnot(None)
                if index_type == "model"
                else ModelEmbedding.dataset_id.isnot(None),
            )
            .order_by(ModelEmbedding.embedding.cosine_distance(qvec))
            .limit(top_k)
            .all()
        )
        results = []
        for row in rows:
            distance = self._pg_distance(db, row, qvec)
            score = max(0.0, 1.0 - distance)
            meta = row.metadata_json or {}
            results.append(self._to_result(meta, score, row.model_id, row.dataset_id))
        return results

    def _search_bruteforce(
        self, db, tenant_id: str, qvec: List[float], index_type: str, top_k: int,
    ) -> List[Dict[str, Any]]:
        q = np.asarray(qvec, dtype=np.float32)
        rows = (
            db.query(ModelEmbedding)
            .filter(ModelEmbedding.tenant_id == tenant_id)
            .all()
        )
        scored = []
        for row in rows:
            meta = row.metadata_json or {}
            if meta.get("index_type") != index_type:
                continue
            vec = np.asarray(row.embedding, dtype=np.float32)
            denom = float(np.linalg.norm(q) * np.linalg.norm(vec)) + 1e-9
            sim = float(np.dot(q, vec) / denom)
            scored.append((sim, meta, row.model_id, row.dataset_id))
        scored.sort(key=lambda x: -x[0])
        return [
            self._to_result(meta, round(sim, 4), model_id, dataset_id)
            for sim, meta, model_id, dataset_id in scored[:top_k]
        ]

    def _to_result(
        self, meta: Dict[str, Any], score: float,
        model_id: Optional[str] = None, dataset_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "index_type": meta.get("index_type"),
            "model_id": model_id or meta.get("model_id"),
            "dataset_id": dataset_id or meta.get("dataset_id"),
            "name": meta.get("model_name") or meta.get("name"),
            "score": score,
            "metadata": meta,
        }

    @staticmethod
    def _pg_distance(db, row, qvec) -> float:
        # `<=>` devuelve la distancia coseno (1 - sim para vectores normalizados).
        # En tests (sqlite) esta rama no se usa.
        try:
            q = np.asarray(qvec, dtype=np.float32)
            v = np.asarray(row.embedding, dtype=np.float32)
            denom = float(np.linalg.norm(q) * np.linalg.norm(v)) + 1e-9
            return float(1.0 - np.dot(q, v) / denom)
        except Exception:
            return 1.0


_embedding_service_instance: Optional[EmbeddingService] = None


def get_embedding_service(use_cache: bool = True) -> EmbeddingService:
    """Factory singleton del EmbeddingService."""
    global _embedding_service_instance
    if _embedding_service_instance is None:
        _embedding_service_instance = EmbeddingService(use_cache=use_cache)
    return _embedding_service_instance
