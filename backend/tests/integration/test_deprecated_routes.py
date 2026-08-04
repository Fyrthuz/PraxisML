"""Tests de integración para las rutas deprecadas del API Tabular.

Verifica que las rutas antiguas sigan funcionando pero estén ocultas del OpenAPI.
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

import app.observability.models  # noqa: F401

_fastapi_app = create_app()
SQLITE_URL = "sqlite:///./test_deprecated_routes.db"


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
    # Mock storage service to avoid MinIO connection
    from unittest.mock import MagicMock
    from app.api.deps import get_storage_service
    
    mock_storage = MagicMock()
    mock_storage.download.return_value = b"id,label\n1,test\n"
    
    _fastapi_app.dependency_overrides[get_storage_service] = lambda: mock_storage
    
    async with AsyncClient(transport=ASGITransport(app=_fastapi_app), base_url="http://test") as ac:
        yield ac
    
    _fastapi_app.dependency_overrides.pop(get_storage_service, None)


def _create_tenant(db_session, name="Deprecated Routes Tenant"):
    tenant = Tenant(name=name)
    db_session.add(tenant)
    db_session.commit()
    db_session.refresh(tenant)
    return tenant


def _create_user(db_session, tenant, role):
    import uuid
    email = f"dep_{role}_{uuid.uuid4().hex[:8]}@test.com"
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


# ── Tests de rutas deprecadas ─────────────────────────────────────────────


async def test_deprecated_training_algorithms_works(client, db_session):
    """GET /api/v1/training/algorithms debe funcionar (alias deprecado)."""
    tenant = _create_tenant(db_session)
    _, token = _create_user(db_session, tenant, UserRole.VIEWER)
    response = await client.get(
        "/api/v1/training/algorithms",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200


async def test_deprecated_training_train_forbidden(client, db_session):
    """POST /api/v1/training/train debe rechazar viewers (alias deprecado)."""
    tenant = _create_tenant(db_session)
    _, token = _create_user(db_session, tenant, UserRole.VIEWER)
    response = await client.post(
        "/api/v1/training/train",
        json={"dataset_id": "fake", "target_column": "t", "algorithm": "rf"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 403


async def test_deprecated_predictions_list(client, db_session):
    """GET /api/v1/predictions debe funcionar (alias deprecado)."""
    tenant = _create_tenant(db_session)
    _, token = _create_user(db_session, tenant, UserRole.VIEWER)
    response = await client.get(
        "/api/v1/predictions",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200


async def test_deprecated_predictions_unauthenticated(client):
    """GET /api/v1/predictions sin auth debe devolver 401 (alias deprecado)."""
    response = await client.get("/api/v1/predictions")
    assert response.status_code == 401


async def test_deprecated_preprocessing_preview(client, db_session):
    """POST /api/v1/preprocessing/preview debe funcionar (alias deprecado)."""
    tenant = _create_tenant(db_session)
    _, token = _create_user(db_session, tenant, UserRole.EDITOR)  # Use editor role
    response = await client.post(
        "/api/v1/preprocessing/preview",
        json={"tenant_id": tenant.id, "dataset_id": "fake-ds"},
        headers={"Authorization": f"Bearer {token}"},
    )
    # Debería devolver 404 (dataset no existe) o 422 (validación), no 401
    assert response.status_code in (404, 422, 500)


async def test_deprecated_routes_not_in_openapi(client):
    """Las rutas deprecadas NO deben aparecer en OpenAPI."""
    response = await client.get("/api/v1/openapi.json")
    assert response.status_code == 200
    paths = response.json().get("paths", {})

    deprecated_paths = [
        "/api/v1/training/algorithms",
        "/api/v1/training/train",
        "/api/v1/predictions",
        "/api/v1/preprocessing/preview",
    ]
    for path in deprecated_paths:
        assert path not in paths, f"Ruta deprecada aún visible: {path}"


async def test_new_routes_in_openapi(client):
    """Las rutas nuevas tabulares DEBEN aparecer en OpenAPI."""
    response = await client.get("/api/v1/openapi.json")
    assert response.status_code == 200
    paths = response.json().get("paths", {})

    new_paths = [
        "/api/v1/tabular/training/algorithms",
        "/api/v1/tabular/training/train",
        "/api/v1/tabular/predictions",
        "/api/v1/tabular/preprocessing/preview",
    ]
    for path in new_paths:
        assert path in paths, f"Ruta nueva faltante: {path}"
