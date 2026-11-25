"""Tests de integración para Workspaces, API Keys, Audit Log, Event Bus y Model Serving Endpoints."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.platform.events.event_bus import DomainEventBus
from app.core.security import create_access_token, get_password_hash
from app.database import get_db
from app.main import app
from app.models.base import Base
from app.models.ml_model import MLModel
from app.models.tenant import Tenant
from app.models.user import User

SQLALCHEMY_TEST_URL = "sqlite:///./test_phase2_enterprise_serving.db"
engine = create_engine(SQLALCHEMY_TEST_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(autouse=True)
def setup_db():
    Base.metadata.create_all(bind=engine)
    app.dependency_overrides[get_db] = override_get_db
    yield
    app.dependency_overrides.pop(get_db, None)
    Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def auth_context():
    db = TestingSessionLocal()
    tenant = Tenant(name="Serving Corp")
    db.add(tenant)
    db.commit()
    db.refresh(tenant)

    user = User(
        email="owner@servingcorp.io",
        hashed_password=get_password_hash("secret123"),
        role="admin",
        tenant_id=tenant.id,
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    model = MLModel(
        name="Titanic RF",
        mlflow_run_id="run-titanic-primary",
        tenant_id=tenant.id,
        version="1.0.0",
        stage="Production",
        metrics_metadata={
            "framework": "sklearn",
            "task_type": "classification",
            "feature_names": ["Age", "Fare", "Pclass"],
        },
    )
    canary = MLModel(
        name="Titanic GBM Canary",
        mlflow_run_id="run-titanic-canary",
        tenant_id=tenant.id,
        version="1.1.0",
        stage="Staging",
        metrics_metadata={
            "framework": "sklearn",
            "task_type": "classification",
            "feature_names": ["Age", "Fare", "Pclass"],
        },
    )
    db.add_all([model, canary])
    db.commit()
    db.refresh(model)
    db.refresh(canary)

    token = create_access_token(subject=user.id)
    ctx = {
        "tenant_id": tenant.id,
        "user_id": user.id,
        "model_id": model.id,
        "canary_id": canary.id,
        "headers": {"Authorization": f"Bearer {token}"},
    }
    db.close()
    return ctx


def test_workspaces_api_lifecycle(auth_context):
    client = TestClient(app)
    headers = auth_context["headers"]

    # Auto-provisions default workspace
    res = client.get("/api/v1/workspaces/", headers=headers)
    assert res.status_code == 200
    assert res.json()["items"][0]["slug"] == "default"

    # Create staging workspace
    created = client.post(
        "/api/v1/workspaces/",
        headers=headers,
        json={"name": "Risk Staging", "environment": "staging", "default_compute_profile": "cpu-heavy"},
    )
    assert created.status_code == 201
    ws = created.json()
    assert ws["slug"] == "risk-staging"

    # Update & delete
    upd = client.patch(
        f"/api/v1/workspaces/{ws['id']}",
        headers=headers,
        json={"budget_limit_usd": 250.0},
    )
    assert upd.status_code == 200
    assert upd.json()["budget_limit_usd"] == 250.0

    deleted = client.delete(f"/api/v1/workspaces/{ws['id']}", headers=headers)
    assert deleted.status_code == 200


def test_api_keys_authentication_and_revocation(auth_context):
    client = TestClient(app)
    headers = auth_context["headers"]

    res = client.post(
        "/api/v1/api-keys/",
        headers=headers,
        json={"name": "ci-serving", "scopes": ["serving:invoke", "models:read"]},
    )
    assert res.status_code == 201
    body = res.json()
    raw_key = body["api_key"]
    assert raw_key.startswith("px_")

    # Authenticate via X-API-Key
    r1 = client.get("/api/v1/serving/endpoints", headers={"X-API-Key": raw_key})
    assert r1.status_code == 200

    # Authenticate via Bearer px_...
    r2 = client.get("/api/v1/serving/endpoints", headers={"Authorization": f"Bearer {raw_key}"})
    assert r2.status_code == 200

    # Revoke and confirm 401
    rev = client.delete(f"/api/v1/api-keys/{body['id']}", headers=headers)
    assert rev.status_code == 200

    r3 = client.get("/api/v1/serving/endpoints", headers={"X-API-Key": raw_key})
    assert r3.status_code == 401


def test_model_endpoint_deploy_canary_and_invoke_with_api_key(auth_context):
    client = TestClient(app)
    headers = auth_context["headers"]

    # Create scoped API key for serving
    key_res = client.post(
        "/api/v1/api-keys/",
        headers=headers,
        json={"name": "serving-only", "scopes": ["serving:invoke"]},
    )
    api_key = key_res.json()["api_key"]

    # Create restricted key without serving:invoke
    no_serve_key = client.post(
        "/api/v1/api-keys/",
        headers=headers,
        json={"name": "read-only", "scopes": ["models:read"]},
    ).json()["api_key"]

    # Deploy endpoint with 100% canary routing for deterministic check
    ep_res = client.post(
        "/api/v1/serving/endpoints",
        headers=headers,
        json={
            "name": "Titanic Live",
            "slug": "titanic-live",
            "model_id": auth_context["model_id"],
            "secondary_model_id": auth_context["canary_id"],
            "traffic_split_pct": 100,
        },
    )
    assert ep_res.status_code == 201
    ep = ep_res.json()
    assert ep["slug"] == "titanic-live"
    assert ep["invoke_path"] == "/serving/v1/titanic-live/invocations"

    # Mock loaded sklearn model
    fake_model = MagicMock()
    fake_model.predict.return_value = np.array([1])
    fake_model.predict_proba.return_value = np.array([[0.15, 0.85]])

    with patch("app.platform.serving.runtimes.get_inference_service") as mock_inf:
        mock_inf.return_value.load_model.return_value = fake_model

        # Key lacking scope must receive 403
        forbidden = client.post(
            "/serving/v1/titanic-live/invocations",
            headers={"X-API-Key": no_serve_key},
            json={"inputs": {"Age": 29, "Fare": 35.0, "Pclass": 1}},
        )
        assert forbidden.status_code == 403

        # Valid serving API key invokes successfully
        inv = client.post(
            "/serving/v1/titanic-live/invocations",
            headers={"X-API-Key": api_key},
            json={"inputs": {"Age": 29, "Fare": 35.0, "Pclass": 1}},
        )
        assert inv.status_code == 200
        out = inv.json()
        assert out["endpoint_slug"] == "titanic-live"
        assert out["variant"] == "canary"
        assert out["model_id"] == auth_context["canary_id"]
        assert out["predictions"] == [1]
        assert out["probabilities"] == [[0.15, 0.85]]

    # Pause blocks invocation with 503
    pause = client.patch(
        f"/api/v1/serving/endpoints/{ep['id']}",
        headers=headers,
        json={"status": "paused"},
    )
    assert pause.status_code == 200
    blocked = client.post(
        "/serving/v1/titanic-live/invocations",
        headers={"X-API-Key": api_key},
        json={"inputs": {"Age": 29, "Fare": 35.0, "Pclass": 1}},
    )
    assert blocked.status_code == 503

    # Audit log captured key creation & endpoint deployment
    audit = client.get("/api/v1/audit/", headers=headers)
    assert audit.status_code == 200
    actions = {row["action"] for row in audit.json()["items"]}
    assert "auth.key_created" in actions
    assert "endpoint.deployed" in actions

    # Invocation logs captured the successful live request
    logs_res = client.get(f"/api/v1/serving/endpoints/{ep['id']}/logs", headers=headers)
    assert logs_res.status_code == 200
    logs = logs_res.json()["items"]
    assert len(logs) == 1
    assert logs[0]["status"] == "ok"
    assert logs[0]["variant"] == "canary"
    assert logs[0]["sample_output"]["predictions"] == [1]

    # Per-endpoint observability metrics (percentiles, variant split, latency series)
    metrics_res = client.get(f"/api/v1/serving/endpoints/{ep['id']}/metrics", headers=headers)
    assert metrics_res.status_code == 200
    ep_metrics = metrics_res.json()
    assert ep_metrics["total_requests"] == 1
    assert ep_metrics["success_rate_pct"] == 100.0
    assert ep_metrics["by_variant"]["canary"]["count"] == 1
    assert ep_metrics["total_rows_scored"] == 1
    assert len(ep_metrics["latency_series"]) == 1

    # Unified platform observability summary reflects deployed endpoints & serving usage
    summary_res = client.get("/api/v1/observability/platform-summary", headers=headers)
    assert summary_res.status_code == 200
    summary = summary_res.json()
    assert summary["serving"]["total_endpoints"] == 1
    assert summary["serving"]["total_requests"] == 1
    assert summary["usage_by_domain"].get("serving", 0) >= 1
    assert "postgres" in summary["services_health"]
    assert "mlflow" in summary["services_health"]


def test_event_bus_local_dispatch_and_drift_helpers():
    bus = DomainEventBus()
    captured = []
    bus.subscribe("endpoint.deployed", lambda ev: captured.append(ev))
    ev = bus.publish("endpoint.deployed", tenant_id="t-1", payload={"slug": "demo"})
    assert len(captured) == 1
    assert captured[0].event_id == ev.event_id

    from app.api.routes.v1.drift import calculate_drift_for_model

    db = TestingSessionLocal()
    try:
        assert calculate_drift_for_model("missing-model", db) is None
    finally:
        db.close()


def test_runtime_environments_and_custom_python_pipeline_execution(auth_context):
    from app.models.dataset import Dataset
    from app.platform.storage.service import get_storage

    client = TestClient(app)
    headers = auth_context["headers"]

    # 1. Capabilities introspection exposes all pluggable registries
    caps = client.get("/api/v1/environments/capabilities", headers=headers)
    assert caps.status_code == 200
    body = caps.json()
    assert "s3" in body["storage_backends"]
    assert "celery" in body["compute_backends"]
    assert "uv_isolated" in body["environment_runners"]
    assert "docker" in body["environment_runners"]
    assert "pyfunc" in body["serving_runtimes"]
    assert "redis" in body["event_bus_backends"]
    assert "pandas" in body["data_engines"]
    assert "mlflow" in body["tracker_backends"]

    # 2. Create custom runtime environment
    env_res = client.post(
        "/api/v1/environments/",
        headers=headers,
        json={
            "name": "Feature Lab Py310",
            "python_version": "3.10",
            "packages": [],
            "compute_profile": "cpu-light",
            "runner_type": "uv_isolated",
        },
    )
    assert env_res.status_code == 201
    env_id = env_res.json()["id"]

    # 3. Seed a small tabular dataset in storage + DB
    csv_key = f"tenants/{auth_context['tenant_id']}/datasets/demo_custom_pipe.csv"
    csv_content = b"Age,Fare,Survived\n20,10.0,0\n40,30.0,1\n"
    storage = get_storage()
    storage.upload(csv_key, csv_content)

    db = TestingSessionLocal()
    try:
        ds = Dataset(
            name="demo_raw",
            file_path=csv_key,
            file_size_bytes=len(csv_content),
            file_type="csv",
            num_rows=2,
            num_columns=3,
            column_names=["Age", "Fare", "Survived"],
            tenant_id=auth_context["tenant_id"],
        )
        db.add(ds)
        db.commit()
        db.refresh(ds)
        ds_id = ds.id
    finally:
        db.close()

    # 4. Create custom Python pipeline bound to that environment
    pipe_res = client.post(
        "/api/v1/pipelines/",
        headers=headers,
        json={
            "name": "Custom Ratio Features",
            "environment_id": env_id,
            "engine": "python",
            "custom_code": (
                "def transform(df):\n"
                "    out = df.copy()\n"
                "    out['fare_per_year'] = out['Fare'] / out['Age']\n"
                "    return out\n"
            ),
            "column_renames": {"Age": "passenger_age"},
        },
    )
    assert pipe_res.status_code == 201
    pipe_id = pipe_res.json()["id"]

    # 5. Execute custom pipeline on the dataset
    run_res = client.post(
        f"/api/v1/pipelines/{pipe_id}/run",
        headers=headers,
        json={"dataset_id": ds_id, "output_dataset_name": "demo_enriched"},
    )
    assert run_res.status_code == 200, run_res.text
    run_out = run_res.json()
    assert run_out["output_dataset_name"] == "demo_enriched"
    assert run_out["transformed_shape"] == [2, 4]
    assert "passenger_age" in run_out["transformed_columns"]
    assert "fare_per_year" in run_out["transformed_columns"]
    storage.delete(csv_key)
