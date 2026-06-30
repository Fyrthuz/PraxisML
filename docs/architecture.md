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
                    │  (metadata)  │   │  (Celery +    │  │ (Tracking │
                    │              │   │   Cache)      │  │+ Registry)│
                    └──────────────┘   └───────────────┘  └────┬──────┘
                                                               │
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
```

## Data Flow

### Training Pipeline

```
1. Dataset Upload ──► StorageService ──► DVC (versioned)
       │
2. Preprocessing ──► build_pipeline() ──► MLflow (saved as artifact)
       │
3. Training ──► SklearnTrainer / PyTorchTrainer ──► MLflow (autolog)
       │
4. Model Registration ──► MLflow Registry + DB (MLModel table)
       │
5. Inference ──► InferenceService ──► Uncertainty estimation ──► Result
```

### Inference Flow

```
Sync:  Client ──► API ──► InferenceService ──► Model (TorchScript/MLflow) ──► Response
Async: Client ──► API ──► Celery Task ──► InferenceService ──► Poll for result
Stream: WebSocket ──► API ──► InferenceService ──► Stream results
```

## Multi-Tenant Isolation

- **Database**: All tables have `tenant_id` foreign key
- **Storage**: Keys prefixed with `tenants/{tenant_id}/`
- **MLflow**: Experiments named `tenant_{id}_training`, models prefixed `tenant_{id}_`
- **DVC**: Separate repository per tenant
- **Auth**: JWT tokens scoped to a single tenant
- **RBAC**: 3 roles per tenant (admin, editor, viewer)

## Key Design Decisions

1. **StorageService abstraction** — Pluggable backends (Local/MinIO/S3) via `STORAGE_BACKEND` env var
2. **Declarative preprocessing** — JSON config → sklearn `ColumnTransformer` pipeline
3. **PyTorch uncertainty** — Extensible via `IUncertaintyAlgorithm` interface
4. **Async training** — Celery workers prevent API blocking
5. **Model cache** — LRU + TTL for inference performance
