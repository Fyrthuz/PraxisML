"""
Tests de integración para la API de inferencia NLP.
SQLite + override de get_db. Servicio de inferencia mockeado.
"""

from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import create_app
from app.database import get_db
from app.models.base import Base
from app.models.user import User, UserRole
from app.models.tenant import Tenant
from app.core.security import create_access_token, get_password_hash

import app.models.embedding  # noqa: F401
import app.models.assistant  # noqa: F401

_fastapi_app = create_app()
SQLITE_URL = "sqlite:///./test_nlp_inference_api.db"


@pytest.fixture(scope="module")
def engine():
    eng = create_engine(SQLITE_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=eng)
    yield eng
    Base.metadata.drop_all(bind=eng)


@pytest.fixture()
def db_session(engine):
    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture(autouse=True)
def _clean_db(engine):
    with engine.connect() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            conn.execute(table.delete())
        conn.commit()
    yield


@pytest.fixture()
def override_db(db_session):
    _fastapi_app.dependency_overrides[get_db] = lambda: db_session
    yield
    _fastapi_app.dependency_overrides.clear()


@pytest_asyncio.fixture()
async def client(override_db):
    async with AsyncClient(
        transport=ASGITransport(app=_fastapi_app), base_url="http://test"
    ) as ac:
        yield ac


def _create_tenant(db_session, name="NLP Infer Tenant"):
    tenant = Tenant(name=name)
    db_session.add(tenant)
    db_session.commit()
    db_session.refresh(tenant)
    return tenant


def _create_user(db_session, tenant, role):
    import uuid

    email = f"nlpi_{role}_{uuid.uuid4().hex[:8]}@test.com"
    user = User(
        email=email,
        hashed_password=get_password_hash("pass"),
        full_name="Test",
        tenant_id=tenant.id,
        role=role,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user, create_access_token(subject=user.id)


def _create_nlp_model(db_session, tenant):
    from app.models.ml_model import MLModel

    ml_model = MLModel(
        name="Sentiment DistilBERT",
        mlflow_run_id="run-abc",
        metrics_metadata={
            "domain": "nlp",
            "framework": "huggingface",
            "algorithm": "distilbert",
            "task_type": "text_classification",
            "label_map": {"neg": 0, "pos": 1},
            "id2label": {"0": "neg", "1": "pos"},
            "max_seq_length": 128,
        },
        is_active=True,
        is_public=False,
        tenant_id=tenant.id,
    )
    db_session.add(ml_model)
    db_session.commit()
    db_session.refresh(ml_model)
    return ml_model


def _create_tabular_model(db_session, tenant):
    from app.models.ml_model import MLModel

    ml_model = MLModel(
        name="Random Forest",
        mlflow_run_id="run-tab",
        metrics_metadata={"domain": "tabular", "framework": "sklearn", "algorithm": "random_forest"},
        is_active=True,
        tenant_id=tenant.id,
    )
    db_session.add(ml_model)
    db_session.commit()
    db_session.refresh(ml_model)
    return ml_model


async def test_nlp_predict_success(client, db_session):
    tenant = _create_tenant(db_session)
    _, token = _create_user(db_session, tenant, UserRole.EDITOR)
    model = _create_nlp_model(db_session, tenant)

    fake_service = MagicMock()
    fake_service.load_model.return_value = {"kind": "onnx"}
    fake_service.predict.return_value = {
        "label": "pos",
        "confidence": 0.93,
        "entities": [],
        "tokens": ["good", "product"],
        "latency_ms": 4.2,
    }

    with patch(
        "app.services.nlp_inference.get_nlp_inference_service",
        return_value=fake_service,
    ):
        response = await client.post(
            "/api/v1/nlp/predict",
            json={"model_id": model.id, "text": "Great product!"},
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "pos"
    assert body["confidence"] == 0.93
    assert body["latency_ms"] == 4.2


async def test_nlp_predict_missing_model(client, db_session):
    tenant = _create_tenant(db_session)
    _, token = _create_user(db_session, tenant, UserRole.EDITOR)
    response = await client.post(
        "/api/v1/nlp/predict",
        json={"model_id": "no-existe", "text": "hola"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 404


async def test_nlp_predict_wrong_domain(client, db_session):
    tenant = _create_tenant(db_session)
    _, token = _create_user(db_session, tenant, UserRole.EDITOR)
    tab_model = _create_tabular_model(db_session, tenant)
    response = await client.post(
        "/api/v1/nlp/predict",
        json={"model_id": tab_model.id, "text": "hola"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 400


async def test_nlp_predict_viewer_forbidden(client, db_session):
    tenant = _create_tenant(db_session)
    _, token = _create_user(db_session, tenant, UserRole.VIEWER)
    model = _create_nlp_model(db_session, tenant)
    response = await client.post(
        "/api/v1/nlp/predict",
        json={"model_id": model.id, "text": "hola"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 403


async def test_nlp_predict_batch_enqueues(client, db_session):
    tenant = _create_tenant(db_session)
    _, token = _create_user(db_session, tenant, UserRole.EDITOR)
    model = _create_nlp_model(db_session, tenant)

    mock_task = MagicMock()
    mock_task.id = "celery-batch-1"
    with patch("app.worker.tasks.predict_nlp.run_nlp_batch_inference") as mock_run:
        mock_run.delay.return_value = mock_task
        response = await client.post(
            "/api/v1/nlp/predict/batch",
            data={"model_id": model.id, "text_column": "text"},
            files={"file": ("batch.csv", b"text\ngreat\nbad\n", "text/csv")},
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 202
    body = response.json()
    assert body["prediction_id"]
    assert body["task_id"] == "celery-batch-1"
    mock_run.delay.assert_called_once()


async def test_nlp_streaming_ws(client, db_session, engine):
    """WS /nlp/streaming/{model_id} autentica con token y responde con predicción."""
    import json as jsonlib

    from fastapi.testclient import TestClient
    from sqlalchemy.orm import sessionmaker

    tenant = _create_tenant(db_session)
    _, token = _create_user(db_session, tenant, UserRole.EDITOR)
    model = _create_nlp_model(db_session, tenant)

    TestSession = sessionmaker(bind=engine)

    fake_service = MagicMock()
    fake_service.load_model.return_value = {"kind": "onnx"}
    fake_service.predict.return_value = {
        "label": "pos", "confidence": 0.9, "entities": [],
        "tokens": ["nice"], "latency_ms": 3.0,
    }

    with patch(
        "app.services.nlp_inference.get_nlp_inference_service",
        return_value=fake_service,
    ), patch("app.database.SessionLocal", TestSession):
        with TestClient(_fastapi_app) as tc:
            with tc.websocket_connect(
                f"/api/v1/nlp/streaming/{model.id}?token={token}"
            ) as ws:
                ws.send_text(jsonlib.dumps({"text": "nice"}))
                data = jsonlib.loads(ws.receive_text())
                assert data["label"] == "pos"
