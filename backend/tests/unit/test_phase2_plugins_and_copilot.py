"""
Tests unitarios de Fase 2.1 / 2.4:
  - Plugin Architecture (PluginRegistry, TabularPlugin, NLPPlugin)
  - CopilotToolRegistry & PolicyEngine (4-Tier RBAC, Budget, Prod Approval)
  - ComputeBackend (CeleryComputeBackend & colas tipadas en celery_app)
  - InferenceService._resolve_torchscript_path (soporte stateless S3/MinIO)
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from app.platform.compute.base import ComputeProfile, JobSpec, get_compute_backend
from app.platform.compute.celery_backend import CeleryComputeBackend
from app.platform.copilot.contracts import (
    ActionRiskTier,
    CopilotTool,
    ExecutionIdentity,
    PolicyDecision,
)
from app.platform.copilot.policy_engine import PolicyEngine
from app.platform.copilot.registry import CopilotToolRegistry
from app.plugins.registry import PluginRegistry
from app.worker.celery_app import celery_app


class _DummyArgs(BaseModel):
    value: str = "ok"


class TestPluginArchitecture:
    def test_discover_builtin_plugins_loads_tabular_and_nlp(self):
        registry = PluginRegistry()
        loaded = registry.discover_builtin_plugins()

        assert "tabular" in loaded
        assert "nlp" in loaded
        assert registry.has("tabular")
        assert registry.has("nlp")
        assert registry.get("tabular").default_queue == "cpu-heavy"
        assert registry.get("nlp").default_queue == "gpu-train"

    def test_plugin_metadata_and_copilot_tool_registration(self):
        plugin_registry = PluginRegistry()
        plugin_registry.discover_builtin_plugins()

        meta = {m["name"]: m for m in plugin_registry.list_metadata()}
        assert meta["tabular"]["algorithms_count"] > 0
        assert meta["nlp"]["algorithms_count"] > 0

        tool_registry = CopilotToolRegistry()
        tool_registry.register_core_platform_tools()
        added = plugin_registry.register_copilot_tools(tool_registry)

        assert added >= 3
        assert tool_registry.has("models.list")
        assert tool_registry.has("datasets.list")
        assert tool_registry.has("tabular.list_algorithms")
        assert tool_registry.has("nlp.list_algorithms")
        assert tool_registry.has("nlp.semantic_search")

    def test_get_unknown_plugin_raises_key_error(self):
        registry = PluginRegistry()
        with pytest.raises(KeyError):
            registry.get("nonexistent_domain")


class TestCopilotPolicyEngine:
    def _make_tool(self, name: str, min_role: str, tier: ActionRiskTier) -> CopilotTool:
        return CopilotTool(
            name=name,
            namespace="test",
            description="Test tool",
            args_schema=_DummyArgs,
            handler=lambda args, **ctx: {"ok": args.value},
            min_role=min_role,
            risk_tier=tier,
            prod_requires_approval=True,
        )

    def test_viewer_allowed_tier0_and_denied_editor_tool(self):
        engine = PolicyEngine(mutate_approval_threshold_usd=5.0)
        viewer = ExecutionIdentity(user_id="u1", tenant_id="t1", role="viewer")

        read_tool = self._make_tool("t.read", "viewer", ActionRiskTier.READ_METADATA)
        write_tool = self._make_tool("t.train", "editor", ActionRiskTier.MUTATE_DEV)

        assert engine.evaluate(read_tool, viewer).decision == PolicyDecision.ALLOW
        assert engine.evaluate(write_tool, viewer).decision == PolicyDecision.DENY

    def test_tier2_allows_cheap_dev_but_requires_approval_in_prod_or_high_cost(self):
        engine = PolicyEngine(mutate_approval_threshold_usd=5.0)
        tool = self._make_tool("t.train", "editor", ActionRiskTier.MUTATE_DEV)

        editor_dev = ExecutionIdentity(
            user_id="u2", tenant_id="t1", role="editor", environment="development"
        )
        editor_prod = ExecutionIdentity(
            user_id="u2", tenant_id="t1", role="editor", environment="production"
        )

        # Dev barato -> ALLOW
        assert (
            engine.evaluate(tool, editor_dev, estimated_cost_usd=1.0).decision
            == PolicyDecision.ALLOW
        )
        # Dev caro (>= threshold) -> REQUIRE_APPROVAL
        assert (
            engine.evaluate(tool, editor_dev, estimated_cost_usd=10.0).decision
            == PolicyDecision.REQUIRE_APPROVAL
        )
        # Prod -> REQUIRE_APPROVAL incluso barato
        assert (
            engine.evaluate(tool, editor_prod, estimated_cost_usd=0.5).decision
            == PolicyDecision.REQUIRE_APPROVAL
        )
        # Excede presupuesto restante -> DENY
        assert (
            engine.evaluate(
                tool, editor_dev, estimated_cost_usd=12.0, budget_remaining_usd=5.0
            ).decision
            == PolicyDecision.DENY
        )

    def test_tier3_always_requires_human_approval_even_for_admin(self):
        engine = PolicyEngine()
        admin = ExecutionIdentity(
            user_id="u3", tenant_id="t1", role="admin", environment="development"
        )
        critical_tool = self._make_tool(
            "models.promote_stage", "editor", ActionRiskTier.CRITICAL_PROD
        )
        decision = engine.evaluate(critical_tool, admin)
        assert decision.decision == PolicyDecision.REQUIRE_APPROVAL

    def test_tool_registry_filters_by_role_and_exports_openai_schema(self):
        registry = CopilotToolRegistry()
        registry.register_core_platform_tools()

        viewer_names = {t.name for t in registry.list_tools(role="viewer")}
        editor_names = {t.name for t in registry.list_tools(role="editor")}

        assert "models.list" in viewer_names
        assert "models.promote_stage" not in viewer_names
        assert "models.promote_stage" in editor_names

        schemas = registry.list_openai_schemas(role="viewer")
        assert len(schemas) == len(viewer_names)
        assert schemas[0]["type"] == "function"
        assert "parameters" in schemas[0]["function"]


class TestComputeBackendAndTypedQueues:
    def test_celery_task_routes_configured_for_typed_queues(self):
        routes = celery_app.conf.task_routes
        assert routes["app.worker.tasks.scheduled_checks.*"]["queue"] == "system-beat"
        assert routes["app.worker.tasks.predict.*"]["queue"] == "cpu-light"
        assert routes["app.worker.tasks.train.*"]["queue"] == "cpu-heavy"
        assert routes["app.worker.tasks.train_nlp.*"]["queue"] == "gpu-train"

    def test_celery_compute_backend_submit_routes_by_profile(self):
        backend = get_compute_backend()
        assert isinstance(backend, CeleryComputeBackend)

        mock_res = MagicMock()
        mock_res.id = "job-123"
        with patch.object(celery_app, "send_task", return_value=mock_res) as mock_send:
            handle = backend.submit(
                JobSpec(
                    task_name="app.worker.tasks.train_nlp.run_nlp_training",
                    kwargs={"dataset_id": "ds-1"},
                    profile=ComputeProfile.GPU_TRAIN,
                )
            )

        assert handle.job_id == "job-123"
        assert handle.queue == "gpu-train"
        mock_send.assert_called_once_with(
            "app.worker.tasks.train_nlp.run_nlp_training",
            args=[],
            kwargs={"dataset_id": "ds-1"},
            queue="gpu-train",
        )


class TestInferenceServiceRemoteStorage:
    def test_resolve_torchscript_from_remote_storage(self, monkeypatch):
        from app.domains.tabular.inference_service import InferenceService

        fake_storage = MagicMock()
        fake_storage.exists.return_value = True
        fake_storage.download.return_value = b"fake-torchscript-bytes"

        monkeypatch.setattr(
            "app.platform.storage.service.get_storage", lambda: fake_storage
        )

        service = InferenceService(use_cache=False)
        key = "tenants/t1/models/remote_unit_test_model.pt"
        resolved = service._resolve_torchscript_path(key)

        assert isinstance(resolved, Path)
        assert resolved.exists()
        assert resolved.read_bytes() == b"fake-torchscript-bytes"
        # Limpieza del fichero cacheado temporal
        resolved.unlink(missing_ok=True)
