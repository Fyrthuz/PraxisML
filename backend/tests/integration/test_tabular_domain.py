"""
Tests de integración para rutas tabulares dominiales (/api/v1/tabular/*).
Verifica que las nuevas rutas funcionen correctamente junto con los aliases de compatibilidad.
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
SQLITE_URL = "sqlite:///./test_tabular_domain.db"


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
    async with AsyncClient(transport=ASGITransport(app=_fastapi_app), base_url="http://test") as ac:
        yield ac


def _create_tenant(db_session, name="Tabular Domain Tenant"):
    tenant = Tenant(name=name)
    db_session.add(tenant)
    db_session.commit()
    db_session.refresh(tenant)
    return tenant


def _create_user(db_session, tenant, role):
    import uuid
    email = f"tab_{role}_{uuid.uuid4().hex[:8]}@test.com"
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


# ── Nuevas rutas dominiales ───────────────────────────────────────────────


async def test_tabular_algorithms(client, db_session):
    """GET /api/v1/tabular/training/algorithms debe funcionar."""
    tenant = _create_tenant(db_session)
    _, token = _create_user(db_session, tenant, UserRole.VIEWER)
    response = await client.get(
        "/api/v1/tabular/training/algorithms",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "algorithms" in data or isinstance(data, list) or "sklearn" in data or "pytorch" in data


async def test_tabular_train_unauthenticated(client):
    """POST /api/v1/tabular/training/train sin auth debe devolver 401."""
    response = await client.post("/api/v1/tabular/training/train", json={})
    assert response.status_code in (401, 422)


async def test_tabular_train_viewer_forbidden(client, db_session):
    """Viewer NO puede entrenar con la nueva ruta dominal."""
    tenant = _create_tenant(db_session)
    _, token = _create_user(db_session, tenant, UserRole.VIEWER)
    response = await client.post(
        "/api/v1/tabular/training/train",
        json={"dataset_id": "fake", "target_column": "t", "algorithm": "rf"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 403


async def test_tabular_predictions_list(client, db_session):
    """GET /api/v1/tabular/predictions debe funcionar para viewer."""
    tenant = _create_tenant(db_session)
    _, token = _create_user(db_session, tenant, UserRole.VIEWER)
    response = await client.get(
        "/api/v1/tabular/predictions",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200


async def test_tabular_predictions_unauthenticated(client):
    """GET /api/v1/tabular/predictions sin auth debe devolver 401."""
    response = await client.get("/api/v1/tabular/predictions")
    assert response.status_code == 401


# ── Compatibilidad con rutas antiguas ─────────────────────────────────────


async def test_compat_training_algorithms(client, db_session):
    """GET /api/v1/training/algorithms sigue funcionando (alias de compatibilidad)."""
    tenant = _create_tenant(db_session)
    _, token = _create_user(db_session, tenant, UserRole.VIEWER)
    response = await client.get(
        "/api/v1/training/algorithms",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200


async def test_compat_predictions_list(client, db_session):
    """GET /api/v1/predictions sigue funcionando (alias de compatibilidad)."""
    tenant = _create_tenant(db_session)
    _, token = _create_user(db_session, tenant, UserRole.VIEWER)
    response = await client.get(
        "/api/v1/predictions",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200


async def test_compat_training_train_forbidden(client, db_session):
    """POST /api/v1/training/train sigue rechazando viewers."""
    tenant = _create_tenant(db_session)
    _, token = _create_user(db_session, tenant, UserRole.VIEWER)
    response = await client.post(
        "/api/v1/training/train",
        json={"dataset_id": "fake", "target_column": "t", "algorithm": "rf"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 403


# ── Verificación OpenAPI ──────────────────────────────────────────────────


async def test_tabular_routes_in_openapi(client):
    """Las rutas tabulares dominiales deben aparecer en OpenAPI."""
    response = await client.get("/api/v1/openapi.json")
    assert response.status_code == 200
    paths = response.json().get("paths", {})
    tabular_paths = [p for p in paths if "/tabular/" in p]
    assert len(tabular_paths) >= 10, f"Solo {len(tabular_paths)} rutas tabulares encontradas"


async def test_deprecated_routes_hidden_from_openapi(client):
    """Las rutas deprecated no deben aparecer en OpenAPI."""
    response = await client.get("/api/v1/openapi.json")
    assert response.status_code == 200
    paths = response.json().get("paths", {})
    deprecated_paths = [p for p in paths if p.startswith("/api/v1/training/") or p.startswith("/api/v1/preprocessing/")]
    assert len(deprecated_paths) == 0, f"Rutas deprecated aún visibles: {deprecated_paths}"
