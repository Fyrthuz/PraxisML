"""
Tests del RAGService: construcción de contexto y sources sin datos crudos.
"""


import pytest

from app.services.rag_service import RAGService


@pytest.fixture
def db():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.models.base import Base

    import app.models  # noqa: F401
    import app.observability.models  # noqa: F401

    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=eng)
    session = sessionmaker(bind=eng)()
    yield session
    session.close()


def _seed(db):
    from app.models.ml_model import MLModel
    from app.models.tenant import Tenant
    from app.observability.models import CostRecord, DriftHistory
    from datetime import datetime, timezone

    tenant = Tenant(name="T")
    db.add(tenant)
    db.commit()
    db.refresh(tenant)

    model = MLModel(
        name="Sentiment Model",
        mlflow_run_id="r1",
        stage="Production",
        metrics_metadata={
            "domain": "nlp",
            "algorithm": "distilbert",
            "task_type": "sentiment_analysis",
            "metrics": {"test_accuracy": 0.91, "test_f1": 0.90},
        },
        is_active=True,
        tenant_id=tenant.id,
    )
    db.add(model)
    db.commit()
    db.refresh(model)

    now = datetime.now(timezone.utc)
    db.add(DriftHistory(
        tenant_id=tenant.id, model_id=model.id, drift_type="psi",
        metrics={"psi": 0.15}, is_anomaly=False, recorded_at=now,
    ))
    db.add(CostRecord(
        tenant_id=tenant.id, model_id=model.id, cost_type="inference",
        amount=2.5, source="r1", period_start=now, period_end=now,
    ))
    db.commit()
    return tenant, model


def test_retrieve_context_includes_models(db):
    tenant, model = _seed(db)
    import app.services.rag_service as rag_mod

    original = rag_mod.RAGService._semantic_search
    rag_mod.RAGService._semantic_search = lambda self, db, tenant_id, q, top_k=5: {
        "models": [], "datasets": []
    }
    try:
        svc = RAGService()
        result = svc.retrieve_context(db, tenant.id, "¿cómo está mi modelo de sentimiento?")
    finally:
        rag_mod.RAGService._semantic_search = original

    assert "modelos" in result["context"]
    assert "Sentiment Model" in result["context"]
    assert len(result["sources"]) >= 1
    assert result["sources"][0]["model_id"] == model.id


def test_system_prompt_forbids_raw_data():
    svc = RAGService()
    assert "NUNCA" in svc.SYSTEM_PROMPT
    assert "metadatos" in svc.SYSTEM_PROMPT
