"""
Tests de integración para la API de entrenamiento NLP (/api/v1/nlp/training).
SQLite + override de get_db. La tarea Celery se mockea.
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
SQLITE_URL = "sqlite:///./test_nlp_training_api.db"


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


def _create_tenant(db_session, name="NLP Tenant"):
    tenant = Tenant(name=name)
    db_session.add(tenant)
    db_session.commit()
    db_session.refresh(tenant)
    return tenant


def _create_user(db_session, tenant, role):
    import uuid

    email = f"nlp_{role}_{uuid.uuid4().hex[:8]}@test.com"
    user = User(
        email=email,
        hashed_password=get_password_hash("pass"),
        full_name=f"Test {role.capitalize()}",
        tenant_id=tenant.id,
        role=role,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    token = create_access_token(subject=user.id)
    return user, token


def _create_dataset(db_session, tenant):
    from app.models.dataset import Dataset

    dataset = Dataset(
        name="reviews.csv",
        description="sentiment dataset",
        file_path=f"tenants/{tenant.id}/datasets/reviews.csv",
        file_type="csv",
        num_rows=100,
        num_columns=2,
        column_names=["text", "label"],
        version=1,
        tenant_id=tenant.id,
    )
    db_session.add(dataset)
    db_session.commit()
    db_session.refresh(dataset)
    return dataset


async def test_nlp_algorithms_viewer(client, db_session):
    """GET /api/v1/nlp/algorithms debe listar las 6 arquitecturas."""
    tenant = _create_tenant(db_session)
    _, token = _create_user(db_session, tenant, UserRole.VIEWER)
    response = await client.get(
        "/api/v1/nlp/algorithms", headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 6
    ids = {d["id"] for d in data}
    assert {"distilbert", "bert-base", "roberta"} <= ids


async def test_nlp_algorithms_requires_auth(client):
    """Sin token → 401."""
    response = await client.get("/api/v1/nlp/algorithms")
    assert response.status_code == 401


async def test_nlp_train_launches_task(client, db_session):
    """POST /api/v1/nlp/train encola la tarea Celery y devuelve task_id."""
    tenant = _create_tenant(db_session)
    _, token = _create_user(db_session, tenant, UserRole.EDITOR)
    dataset = _create_dataset(db_session, tenant)

    mock_task = MagicMock()
    mock_task.id = "celery-task-123"
    with patch("app.worker.tasks.train_nlp.run_nlp_training") as mock_run:
        mock_run.delay.return_value = mock_task
        response = await client.post(
            "/api/v1/nlp/train",
            json={
                "dataset_id": dataset.id,
                "text_column": "text",
                "label_column": "label",
                "algorithm": "distilbert",
                "task_type": "text_classification",
            },
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["task_id"] == "celery-task-123"
    assert body["status_url"] == "/api/v1/nlp/training/status/celery-task-123"
    mock_run.delay.assert_called_once()
    call_kwargs = mock_run.delay.call_args.kwargs
    assert call_kwargs["text_column"] == "text"
    assert call_kwargs["algorithm"] == "distilbert"
    assert call_kwargs["tenant_id"] == tenant.id


async def test_nlp_train_viewer_forbidden(client, db_session):
    """Viewer no puede lanzar entrenamiento → 403."""
    tenant = _create_tenant(db_session)
    _, token = _create_user(db_session, tenant, UserRole.VIEWER)
    dataset = _create_dataset(db_session, tenant)
    response = await client.post(
        "/api/v1/nlp/train",
        json={
            "dataset_id": dataset.id,
            "text_column": "text",
            "label_column": "label",
            "algorithm": "distilbert",
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 403


async def test_nlp_train_invalid_algorithm(client, db_session):
    """Algoritmo no registrado → 400."""
    tenant = _create_tenant(db_session)
    _, token = _create_user(db_session, tenant, UserRole.EDITOR)
    dataset = _create_dataset(db_session, tenant)
    response = await client.post(
        "/api/v1/nlp/train",
        json={
            "dataset_id": dataset.id,
            "text_column": "text",
            "label_column": "label",
            "algorithm": "gpt-neo",
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 400


async def test_nlp_train_missing_dataset(client, db_session):
    """Dataset inexistente → 404."""
    tenant = _create_tenant(db_session)
    _, token = _create_user(db_session, tenant, UserRole.EDITOR)
    response = await client.post(
        "/api/v1/nlp/train",
        json={
            "dataset_id": "does-not-exist",
            "text_column": "text",
            "label_column": "label",
            "algorithm": "distilbert",
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 404


async def test_nlp_training_status(client, db_session):
    """GET /api/v1/nlp/training/status/{id} devuelve el estado de la tarea."""
    tenant = _create_tenant(db_session)
    _, token = _create_user(db_session, tenant, UserRole.VIEWER)

    with patch("celery.result.AsyncResult") as mock_async:
        mock_result = MagicMock()
        mock_result.status = "SUCCESS"
        mock_result.ready.return_value = True
        mock_result.successful.return_value = True
        mock_result.result = {"status": "COMPLETED", "model_id": "m1"}
        mock_async.return_value = mock_result

        response = await client.get(
            "/api/v1/nlp/training/status/task-xyz",
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "SUCCESS"
    assert body["result"]["model_id"] == "m1"
