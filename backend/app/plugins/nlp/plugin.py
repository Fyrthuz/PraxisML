"""
Plugin del dominio NLP sobre el contrato DomainPlugin.
Encapsula routers NLP, NLPTrainer, NLPInferenceService, NLPProfiler y tools del Copilot.
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


class _NLPAlgorithmsArgs(BaseModel):
    pass


class _SemanticSearchArgs(BaseModel):
    query: str = Field(..., description="Texto de búsqueda semántica sobre modelos y datasets")
    top_k: int = Field(default=5, ge=1, le=20)
    index: str = Field(default="all", description="all | models | datasets")


class NLPTrainerAdapter(BaseTrainer):
    """Adaptador unificado sobre NLPTrainer."""

    domain = "nlp"

    def supported_algorithms(self) -> Dict[str, Dict[str, Any]]:
        from app.core_ml.hyperparams import NLP_ALGORITHMS

        return NLP_ALGORITHMS

    def train(self, ctx: TrainingContext) -> TrainingResult:
        from app.domains.nlp.trainer import NLPTrainer

        df = ctx.extra.get("dataframe")
        text_column = ctx.extra.get("text_column")
        label_column = ctx.extra.get("label_column")
        if df is None or not text_column:
            raise ValueError(
                "NLPTrainerAdapter requiere 'dataframe' y 'text_column' en ctx.extra"
            )

        trainer = NLPTrainer(tenant_id=ctx.tenant_id)
        raw = trainer.train(
            df=df,
            text_column=text_column,
            label_column=label_column,
            algorithm=ctx.algorithm,
            task_type=ctx.task_type,
            hyperparams=ctx.hyperparams,
            validation_config=ctx.validation_config,
            model_name=ctx.extra.get("model_name"),
        )
        return TrainingResult(
            mlflow_run_id=raw["mlflow_run_id"],
            metrics=raw.get("metrics", {}),
            artifact_keys={"onnx": raw.get("onnx_path") or ""},
            duration_s=raw.get("duration_s", 0.0),
            framework="huggingface",
            metadata={
                "label_map": raw.get("label_map"),
                "device": raw.get("device"),
            },
        )


class NLPInferenceAdapter(BaseInferenceEngine):
    """Adaptador unificado sobre NLPInferenceService."""

    domain = "nlp"

    def load(self, model_record: Any, storage: Optional[Any] = None) -> Any:
        return model_record

    def predict(self, loaded_model: Any, payload: Any, **kwargs) -> Dict[str, Any]:
        from app.domains.nlp.inference import get_nlp_inference_service

        svc = get_nlp_inference_service()
        text = payload if isinstance(payload, str) else payload.get("text", "")
        return svc.predict(loaded_model, text)


def _tool_list_nlp_algorithms(_args: _NLPAlgorithmsArgs, **_ctx: Any) -> Dict[str, Any]:
    from app.core_ml.hyperparams import NLP_ALGORITHMS

    return {
        "domain": "nlp",
        "algorithms": {
            k: {
                "display_name": v.get("display_name", k),
                "hf_model": v.get("hf_model"),
                "supported_tasks": v.get("supported_tasks", []),
            }
            for k, v in NLP_ALGORITHMS.items()
        },
    }


def _tool_semantic_search(args: _SemanticSearchArgs, **ctx: Any) -> Dict[str, Any]:
    from app.domains.nlp.embedding_service import get_embedding_service

    db = ctx.get("db")
    identity = ctx.get("identity")
    if db is None or identity is None:
        raise ValueError("Contexto db e identity requeridos para nlp.semantic_search")
    svc = get_embedding_service()
    results = svc.search(
        db=db,
        tenant_id=identity.tenant_id,
        query=args.query,
        top_k=args.top_k,
        index=args.index,
    )
    return {"query": args.query, "results": results}


class NLPPlugin(DomainPlugin):
    name = "nlp"
    version = "1.0.0"
    required_extra = "nlp"
    default_queue = "gpu-train"

    def get_routers(self) -> List[RouterMount]:
        from app.api.routes.v1.nlp import (
            assistant as nlp_assistant,
            embeddings as nlp_embeddings,
            inference as nlp_inference,
            training as nlp_training,
        )

        return [
            RouterMount(nlp_training.router, "/nlp", ["NLP"]),
            RouterMount(nlp_inference.router, "/nlp", ["NLP"]),
            RouterMount(nlp_embeddings.router, "/nlp", ["NLP"]),
            RouterMount(nlp_assistant.router, "/nlp", ["NLP"]),
        ]

    def get_trainer(self) -> BaseTrainer:
        return NLPTrainerAdapter()

    def get_inference_engine(self) -> BaseInferenceEngine:
        return NLPInferenceAdapter()

    def get_profiler(self) -> Optional[Any]:
        from app.domains.nlp.profiler import profile_text_columns

        return profile_text_columns

    def get_copilot_tools(self) -> List[CopilotTool]:
        return [
            CopilotTool(
                name="nlp.list_algorithms",
                namespace="nlp",
                description="Lista las arquitecturas NLP disponibles (DistilBERT, BERT, RoBERTa, XLM-R, BioBERT).",
                args_schema=_NLPAlgorithmsArgs,
                handler=_tool_list_nlp_algorithms,
                min_role="viewer",
                risk_tier=ActionRiskTier.READ_METADATA,
                prod_requires_approval=False,
            ),
            CopilotTool(
                name="nlp.semantic_search",
                namespace="nlp",
                description="Búsqueda semántica sobre modelos y datasets del tenant usando embeddings pgvector.",
                args_schema=_SemanticSearchArgs,
                handler=_tool_semantic_search,
                min_role="viewer",
                risk_tier=ActionRiskTier.SAFE_COMPUTE,
                prod_requires_approval=False,
            ),
        ]
