"""
RAG Service — construye contexto a partir de los metadatos internos de PraxisML
(modelos, drift, costes, búsqueda semántica) para el chatbot asistente.

IMPORTANTE: el LLM solo recibe metadatos; nunca datos crudos de los datasets.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class RAGService:
    """Recupera metadatos internos y los empaqueta como contexto para el LLM."""

    SYSTEM_PROMPT = (
        "Eres el asistente de la plataforma PraxisML. Responde en español de forma "
        "concisa y precisa. SOLO tienes acceso a los metadatos de la organización "
        "proporcionados en el contexto (modelos, métricas, drift, costes). "
        "NUNCA inventes datos ni muestres contenido crudo de los datasets de los "
        "usuarios. Si no tienes información suficiente, dilo claramente."
    )

    def retrieve_context(
        self, db: Session, tenant_id: str, question: str, top_k: int = 5
    ) -> Dict[str, Any]:
        """Recupera el contexto relevante para la pregunta del usuario."""
        models = self._get_models(db, tenant_id)
        drift = self._get_recent_drift(db, tenant_id)
        costs = self._get_recent_costs(db, tenant_id)
        semantic = self._semantic_search(db, tenant_id, question, top_k)

        context = {
            "modelos": models,
            "drift_reciente": drift,
            "costes_recientes": costs,
            "busqueda_semantica": semantic,
        }

        return {
            "context": json.dumps(context, ensure_ascii=False, default=str),
            "sources": self._extract_sources(models, semantic),
        }

    def _get_models(self, db: Session, tenant_id: str) -> List[Dict[str, Any]]:
        from app.models.ml_model import MLModel

        rows = (
            db.query(MLModel)
            .filter(MLModel.tenant_id == tenant_id, MLModel.is_active.is_(True))
            .limit(20)
            .all()
        )
        result = []
        for m in rows:
            meta = m.metrics_metadata or {}
            metrics = meta.get("metrics", {}) or {}
            result.append({
                "model_id": m.id,
                "name": m.name,
                "stage": m.stage,
                "version": m.version,
                "algorithm": meta.get("algorithm"),
                "task_type": meta.get("task_type"),
                "domain": meta.get("domain"),
                "framework": meta.get("framework"),
                "text_column": meta.get("text_column"),
                "label_column": meta.get("label_column"),
                "accuracy": metrics.get("test_accuracy"),
                "f1": metrics.get("test_f1"),
                "precision": metrics.get("test_precision"),
                "recall": metrics.get("test_recall"),
            })
        return result

    def _get_recent_drift(self, db: Session, tenant_id: str) -> List[Dict[str, Any]]:
        from app.observability.models import DriftHistory

        try:
            rows = (
                db.query(DriftHistory)
                .filter(DriftHistory.tenant_id == tenant_id)
                .order_by(DriftHistory.recorded_at.desc())
                .limit(10)
                .all()
            )
            return [
                {
                    "model_id": r.model_id,
                    "drift_type": r.drift_type,
                    "is_anomaly": r.is_anomaly,
                    "metrics": r.metrics,
                    "recorded_at": str(r.recorded_at),
                }
                for r in rows
            ]
        except Exception as exc:
            logger.warning("Drift retrieval failed (non-critical): %s", exc)
            return []

    def _get_recent_costs(self, db: Session, tenant_id: str) -> List[Dict[str, Any]]:
        from app.observability.models import CostRecord

        try:
            rows = (
                db.query(CostRecord)
                .filter(CostRecord.tenant_id == tenant_id)
                .order_by(CostRecord.period_start.desc())
                .limit(10)
                .all()
            )
            return [
                {
                    "cost_type": r.cost_type,
                    "amount": r.amount,
                    "currency": r.currency,
                    "model_id": r.model_id,
                    "source": r.source,
                    "period_start": str(r.period_start),
                }
                for r in rows
            ]
        except Exception as exc:
            logger.warning("Cost retrieval failed (non-critical): %s", exc)
            return []

    def _semantic_search(
        self, db: Session, tenant_id: str, question: str, top_k: int
    ) -> List[Dict[str, Any]]:
        try:
            from app.services.embedding_service import get_embedding_service

            service = get_embedding_service()
            models = service.search(db, tenant_id, question, top_k=top_k, index="models")
            datasets = service.search(db, tenant_id, question, top_k=top_k, index="datasets")
            return {"models": models, "datasets": datasets}
        except Exception as exc:
            logger.warning("Semantic search failed (non-critical): %s", exc)
            return {"models": [], "datasets": []}

    def _extract_sources(self, models: List[Dict], semantic: Dict) -> List[Dict]:
        sources = [
            {"model_id": m.get("model_id"), "name": m.get("name"), "type": "model"}
            for m in models
            if m.get("model_id")
        ]
        for m in semantic.get("models", [])[:3]:
            if m.get("model_id") not in {s["model_id"] for s in sources}:
                sources.append(
                    {"model_id": m.get("model_id"), "name": m.get("name"), "type": "search"}
                )
        return sources


_rag_service_instance: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """Factory singleton del RAGService."""
    global _rag_service_instance
    if _rag_service_instance is None:
        _rag_service_instance = RAGService()
    return _rag_service_instance
