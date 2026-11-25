"""
Plugin del dominio Tabular sobre el contrato DomainPlugin.
Encapsula routers tabulares, trainers, motor de inferencia y tools del Copilot.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.platform.copilot.contracts import ActionRiskTier, CopilotTool
from app.plugins.contracts import (
    BaseInferenceEngine,
    BaseTrainer,
    DomainPlugin,
    RouterMount,
    TrainingContext,
    TrainingResult,
)


class _EmptyArgs(BaseModel):
    pass


class _TabularAlgorithmsArgs(BaseModel):
    task_type: Optional[str] = Field(
        default=None, description="Filtrar por tipo de tarea: classification | regression"
    )


class TabularTrainerAdapter(BaseTrainer):
    """Adaptador unificado sobre SklearnTrainer y PyTorchTrainer."""

    domain = "tabular"

    def supported_algorithms(self) -> Dict[str, Dict[str, Any]]:
        from app.core_ml.hyperparams import ALGORITHM_REGISTRY

        return ALGORITHM_REGISTRY

    def train(self, ctx: TrainingContext) -> TrainingResult:
        from app.core_ml.hyperparams import get_algorithm_info
        from app.domains.tabular.training_service import PyTorchTrainer, SklearnTrainer

        algo_info = get_algorithm_info(ctx.algorithm)
        framework = algo_info.get("framework", "sklearn")
        df = ctx.extra.get("dataframe")
        target_column = ctx.extra.get("target_column")
        if df is None or not target_column:
            raise ValueError(
                "TabularTrainerAdapter requiere 'dataframe' y 'target_column' en ctx.extra"
            )

        trainer = (
            PyTorchTrainer(tenant_id=ctx.tenant_id)
            if framework == "pytorch"
            else SklearnTrainer(tenant_id=ctx.tenant_id)
        )
        raw = trainer.train(
            df=df,
            target_column=target_column,
            algorithm=ctx.algorithm,
            task_type=ctx.task_type,
            hyperparams=ctx.hyperparams,
            validation_config=ctx.validation_config,
            model_name=ctx.extra.get("model_name"),
        )
        return TrainingResult(
            mlflow_run_id=raw["mlflow_run_id"],
            metrics=raw.get("metrics", {}),
            artifact_keys={"model": raw.get("model_path", "")},
            duration_s=raw.get("duration_s", 0.0),
            framework=framework,
            metadata={"feature_names": raw.get("feature_names", [])},
        )


class TabularInferenceAdapter(BaseInferenceEngine):
    """Adaptador unificado sobre InferenceService."""

    domain = "tabular"

    def load(self, model_record: Any, storage: Optional[Any] = None) -> Any:
        from app.domains.tabular.inference_service import InferenceService

        return InferenceService().load_model(model_record)

    def predict(self, loaded_model: Any, payload: Any, **kwargs) -> Dict[str, Any]:
        from app.domains.tabular.inference_service import InferenceService

        method = kwargs.get("method", "ensemble")
        return InferenceService().run_inference(
            model=loaded_model, input_data=payload, method=method
        )


def _tool_list_tabular_algorithms(
    args: _TabularAlgorithmsArgs, **_ctx: Any
) -> Dict[str, Any]:
    from app.core_ml.hyperparams import ALGORITHM_REGISTRY

    items = {}
    for key, info in ALGORITHM_REGISTRY.items():
        supported = info.get("task_types", [])
        if args.task_type and args.task_type not in supported:
            continue
        items[key] = {
            "display_name": info.get("display_name", key),
            "framework": info.get("framework", "sklearn"),
            "task_types": supported,
        }
    return {"domain": "tabular", "algorithms": items}


class TabularPlugin(DomainPlugin):
    name = "tabular"
    version = "1.0.0"
    required_extra = "tabular"
    default_queue = "cpu-heavy"

    def get_routers(self) -> List[RouterMount]:
        from app.api.routes.v1.tabular import (
            predictions as tabular_predictions,
            preprocessing as tabular_preprocessing,
            streaming as tabular_streaming,
            training as tabular_training,
        )

        return [
            # Rutas primarias del dominio Tabular
            RouterMount(tabular_predictions.router, "/tabular", ["Tabular"]),
            RouterMount(
                tabular_preprocessing.router, "/tabular/preprocessing", ["Tabular"]
            ),
            RouterMount(tabular_training.router, "/tabular/training", ["Tabular"]),
            RouterMount(tabular_streaming.router, "/tabular", ["Tabular"]),
            # Aliases de compatibilidad (deprecated, ocultos en OpenAPI)
            RouterMount(
                tabular_predictions.router,
                "",
                ["Tabular (Deprecated)"],
                include_in_schema=False,
            ),
            RouterMount(
                tabular_preprocessing.router,
                "/preprocessing",
                ["Tabular (Deprecated)"],
                include_in_schema=False,
            ),
            RouterMount(
                tabular_training.router,
                "/training",
                ["Tabular (Deprecated)"],
                include_in_schema=False,
            ),
            RouterMount(
                tabular_streaming.router,
                "",
                ["Tabular (Deprecated)"],
                include_in_schema=False,
            ),
        ]

    def get_trainer(self) -> BaseTrainer:
        return TabularTrainerAdapter()

    def get_inference_engine(self) -> BaseInferenceEngine:
        return TabularInferenceAdapter()

    def get_profiler(self) -> Optional[Any]:
        from app.domains.data.data_profiler import profile_dataframe

        return profile_dataframe

    def get_copilot_tools(self) -> List[CopilotTool]:
        return [
            CopilotTool(
                name="tabular.list_algorithms",
                namespace="tabular",
                description="Lista los algoritmos de ML tabular soportados (Scikit-learn y PyTorch MLP).",
                args_schema=_TabularAlgorithmsArgs,
                handler=_tool_list_tabular_algorithms,
                min_role="viewer",
                risk_tier=ActionRiskTier.READ_METADATA,
                prod_requires_approval=False,
            ),
        ]
