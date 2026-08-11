# Architecture Overview

## System Components

```
┌─────────────┐     ┌─────────────────────────────────────────────────────┐
│   Frontend  │     │                   Backend API                       │
│  (Next.js)  │◄───►│              FastAPI + Celery                       │
│             │     │                                                     │
│  Port 3000  │     │  ┌────────────┐  ┌────────────┐  ┌──────────────┐  │
└─────────────┘     │  │ REST API   │  │ WebSocket  │  │ Celery       │  │
                    │  │ Port 8000  │  │ Port 8000  │  │ Workers      │  │
                    │  └─────┬──────┘  └──────┬─────┘  └──────┬───────┘  │
                    │        │                 │               │          │
                    └────────┼─────────────────┼───────────────┼──────────┘
                             │                 │               │
                    ┌────────▼─────┐   ┌───────▼───────┐  ┌────▼──────┐
                    │  PostgreSQL  │   │     Redis     │  │  MLflow   │
                    │  + pgvector  │   │  (Celery +    │  │ (Tracking │
                    │ (metadata +  │   │   Cache)      │  │+ Registry)│
                    │  embeddings) │   └───────────────┘  └────┬──────┘
                    └──────────────┘                           │
                    ┌───────────────────────────────────────────┼──────────┐
                    │           Storage Layer                   │          │
                    │  ┌─────────┐  ┌─────────┐  ┌─────────┐   │          │
                    │  │  Local  │  │  MinIO  │  │   S3    │   │          │
                    │  │ (dev)   │  │(self-   │  │ (AWS)   │   │          │
                    │  │         │  │ hosted) │  │         │   │          │
                    │  └─────────┘  └─────────┘  └─────────┘   │          │
                    └───────────────────────────────────────────┼──────────┘
                                                               │
                                                  ┌────────────▼───────┐
                                                  │   DVC (Data       │
                                                  │   Versioning)     │
                                                  │   MinIO/S3 remote │
                                                  └───────────────────┘

  ┌───────────────┐  (Opcional, profile "nlp", requiere GPU)
  │ vLLM Server   │──► API OpenAI-compatible (/v1) → LLM Gateway (asistente RAG)
  │  Port 8001    │
  └───────────────┘
```

## NLP Vertical (Fase 1)

Router único `/api/v1/nlp/*` con cuatro submódulos:

```
/api/v1/nlp
├── training.py     → GET /algorithms · POST /train · GET /training/status/{task_id}
├── inference.py    → POST /predict · POST /predict/batch · WS /streaming/{model_id}
├── embeddings.py   → POST /embed · GET /search
└── assistant.py    → POST /assistant/chat (JSON/SSE) · GET/DELETE /assistant/history
```

- **NLP Clásico**: `NLPTrainer` (HF Transformers) → export ONNX (`torch.onnx.export`) → `NLPInferenceService` (ONNX Runtime en CPU).
- **Semantic Search**: `EmbeddingService` (sentence-transformers, 384d) → tabla `model_embeddings` (pgvector, índice IVFFlat) → búsqueda `<=>` coseno.
- **Generativo**: `LLMGateway` (vLLM default, OpenAI-compatible) + `RAGService` (contexto solo-metadatos) → asistente con SSE.
- **Observabilidad**: métricas `praxisml_nlp_*` y `praxisml_llm_*`, dashboard Grafana `praxisml-nlp`.

## Data Flow

### Training Pipeline

```
1. Dataset Upload ──► StorageService ──► DVC (versioned)
       │
2. Preprocessing ──► build_pipeline() ──► MLflow (saved as artifact)
       │
3. Training ──► SklearnTrainer / PyTorchTrainer / NLPTrainer ──► MLflow (autolog)
       │
4. Model Registration ──► MLflow Registry + DB (MLModel table)
       │
5. Inference ──► InferenceService / NLPInferenceService ──► Uncertainty estimation ──► Result
```

### NLP Pipeline (Fase 1)

```
1. Upload CSV ──► NLP Profiler hook ──► column_types_analysis + index_dataset (embedding)
       │
2. Train ──► NLPTrainer (HF Trainer) ──► MLflow (tenant_{id}_nlp_training) + export ONNX
       │
3. Index ──► EmbeddingService.index_model ──► model_embeddings (pgvector)
       │
4. Inference ──► ONNX Runtime / transformers ──► {label, confidence, entities[], tokens[], latency_ms}
       │
5. Chat ──► RAGService (metadatos) ──► LLMGateway (vLLM) ──► SSE al frontend
```

### Inference Flow

```
Sync:  Client ──► API ──► InferenceService ──► Model (TorchScript/MLflow) ──► Response
Async: Client ──► API ──► Celery Task ──► InferenceService ──► Poll for result
Stream: WebSocket ──► API ──► InferenceService ──► Stream results
LLM:    Client ──► API ──► RAGService + LLMGateway ──► SSE stream
```

## Multi-Tenant Isolation

- **Database**: All tables have `tenant_id` foreign key
- **Storage**: Keys prefixed with `tenants/{tenant_id}/`
- **MLflow**: Experiments named `tenant_{id}_training` / `tenant_{id}_nlp_training`, models prefixed `tenant_{id}_`
- **DVC**: Separate repository per tenant
- **Embeddings**: `model_embeddings` filtered by `tenant_id` + semantic search returns only tenant rows
- **Auth**: JWT tokens scoped to a single tenant
- **RBAC**: 3 roles per tenant (admin, editor, viewer)

## Key Design Decisions

1. **StorageService abstraction** — Pluggable backends (Local/MinIO/S3) via `STORAGE_BACKEND` env var
2. **Declarative preprocessing** — JSON config → sklearn `ColumnTransformer` pipeline
3. **PyTorch uncertainty** — Extensible via `IUncertaintyAlgorithm` interface
4. **Async training** — Celery workers prevent API blocking
5. **Model cache** — LRU + TTL for inference performance
6. **ONNX export nativo** — `torch.onnx.export` (sin `optimum`, incompatible con torch ≥2.9) → inferencia CPU con `onnxruntime`
7. **VectorType dual** — `vector(384)` en PostgreSQL (pgvector) / `JSON` en SQLite para poder testear sin servidor pgvector
8. **LLM via OpenAI-compatible** — vLLM/OpenAI/Ollama comparten el SDK `openai` con distinta `base_url`; Anthropic usa SDK nativo
9. **RAG solo-metadatos** — el LLM nunca recibe datos crudos de los datasets (privacidad multi-tenant)
