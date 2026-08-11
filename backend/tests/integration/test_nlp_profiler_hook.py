"""
Tests de integración del hook NLP Profiler en POST /datasets/.
SQLite + override de get_db. Se mockea el embedding service del hook.
"""


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
from app.api.deps import get_storage_service

import app.models.embedding  # noqa: F401
import app.models.assistant  # noqa: F401

_fastapi_app = create_app()
SQLITE_URL = "sqlite:///./test_nlp_profiler_hook.db"


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


@pytest.fixture(autouse=True)
def mock_index_dataset(monkeypatch):
    """Evita descargar el modelo de embeddings en el hook."""
    from unittest.mock import MagicMock

    fake_svc = MagicMock()
    monkeypatch.setattr(
        "app.services.embedding_service.get_embedding_service",
        lambda: fake_svc,
    )
    return fake_svc


class FakeStorage:
    def upload(self, key, content, *args, **kwargs):
        self.key = key

    def download(self, key):
        return CSV_TEXTUAL.encode()


@pytest.fixture(autouse=True)
def override_storage():
    _fastapi_app.dependency_overrides[get_storage_service] = lambda: FakeStorage()
    yield
    _fastapi_app.dependency_overrides.pop(get_storage_service, None)


def _create_tenant(db_session, name="Profiler Tenant"):
    tenant = Tenant(name=name)
    db_session.add(tenant)
    db_session.commit()
    db_session.refresh(tenant)
    return tenant


def _create_user(db_session, tenant, role):
    import uuid

    email = f"pr_{role}_{uuid.uuid4().hex[:8]}@test.com"
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


CSV_TEXTUAL = (
    "review,rating,product_id\n"
    "This product is absolutely amazing I love it,5,A1\n"
    "Terrible experience would never buy this again,1,B2\n"
    "Excellent quality and fast shipping highly recommended,5,C3\n"
    "Waste of money completely disappointed,1,A1\n"
    "Great value works as described perfectly,4,B2\n"
    "This product is absolutely amazing I love it,5,C3\n"
    "Terrible experience would never buy this again,1,A1\n"
    "Excellent quality and fast shipping highly recommended,5,B2\n"
    "Waste of money completely disappointed,1,C3\n"
    "Great value works as described perfectly,4,A1\n"
)


async def test_upload_runs_profiler(client, db_session, mock_index_dataset):
    tenant = _create_tenant(db_session)
    _, token = _create_user(db_session, tenant, UserRole.EDITOR)

    response = await client.post(
        "/api/v1/datasets/",
        data={"name": "reviews.csv", "description": "sentiment"},
        files={"file": ("reviews.csv", CSV_TEXTUAL.encode(), "text/csv")},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 201
    body = response.json()
    analysis = body.get("column_types_analysis")
    assert analysis is not None
    assert "review" in analysis["text_columns"]
    assert analysis["suggested_task"] == "sentiment_analysis"
    assert analysis["suggested_algorithm"] == "distilbert"
    assert mock_index_dataset.index_dataset.called
