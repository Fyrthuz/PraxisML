# Setup Guide

## Prerequisites

- Python >= 3.10
- [uv](https://docs.astral.sh/uv/) package manager
- Docker & Docker Compose (for DB, Redis, MinIO, MLflow)
- Node.js >= 18 (for frontend development)

## Quick Start (Docker)

```bash
# 1. Clone & configure
cp .env.example .env

# 2. Start all services
docker compose up --build

# 3. Access services
# Frontend:   http://localhost:3000
# API docs:   http://localhost:8000/docs
# MLflow:     http://localhost:5000
# Grafana:    http://localhost:3001 (admin/admin)
# Prometheus: http://localhost:9090
# MinIO:      http://localhost:9001 (minioadmin/minioadmin)
```

## Development Setup

### 1. Start infrastructure services

```bash
docker compose up -d db redis mlflow minio minio-init
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit with localhost URLs (not Docker hostnames)
```

Or use a custom env file path:

```bash
export PRAXISML_ENV_FILE=/path/to/custom/.env
```

### 3. Backend setup

```bash
cd backend
uv sync --extra dev
uv run alembic upgrade head
uv run uvicorn app.main:app --reload
```

### 4. Celery worker (separate terminal)

```bash
cd backend
uv run celery -A app.worker.celery_app worker --loglevel=info --pool=solo
```

### 5. Run tests

```bash
cd backend
uv run pytest tests/unit/ -v
uv run pytest tests/integration/ -v
# Full coverage report
uv run pytest tests/ --cov=app --cov-report=term-missing
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENVIRONMENT` | `development` | Runtime environment |
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:5432/praxisml_db` | PostgreSQL connection string |
| `SECRET_KEY` | *(insecure default)* | JWT signing key (must be changed in production) |
| `STORAGE_BACKEND` | `local` | Storage backend: `local`, `minio`, or `s3` |
| `PRAXISML_ENV_FILE` | `.env` | Custom path to env file |
| `DVC_REMOTE_NAME` | `minio` | DVC remote name for dataset versioning |
| `CELERY_BROKER_URL` | `redis://localhost:6379/0` | Redis broker for Celery |
| `MLFLOW_TRACKING_URI` | `""` | MLflow tracking server URI |
| `LLM_PROVIDER` | `vllm` | LLM provider: `vllm`, `openai`, `anthropic`, `ollama` |
| `VLLM_BASE_URL` | `http://vllm:8001/v1` | vLLM OpenAI-compatible endpoint |
| `VLLM_MODEL_NAME` | `Qwen/Qwen2.5-7B-Instruct` | Model served by vLLM |
| `LLM_API_KEY` | `EMPTY` | API key (vLLM uses `EMPTY`) |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Embeddings model |

### NLP Engine & vLLM (opcional)

Para el asistente LLM con vLLM (requiere GPU):
```bash
docker compose --profile nlp up -d vllm
export LLM_PROVIDER=vllm
export VLLM_BASE_URL=http://localhost:8001/v1
```

La base de datos usa la imagen `pgvector/pgvector:pg15`; la extensión `vector` se crea automáticamente al aplicar la migración `phase1_nlp_001`.

Generate a secure SECRET_KEY:

```bash
python -c "import secrets; print(secrets.token_urlsafe(64))"
```

## Project Structure

```
backend/
├── app/
│   ├── api/routes/v1/     # REST API endpoints (auth, datasets, models, training, nlp/, etc.)
│   ├── core/              # Config, security, logging, middleware
│   ├── core_ml/           # ML engine (preprocessing, uncertainty, explainability, nlp/)
│   ├── models/            # SQLAlchemy ORM models (incl. embedding, assistant)
│   ├── schemas/           # Pydantic request/response schemas
│   ├── platform/          # 7 pluggable registries (storage, compute, serving, data_engine, tracker, events, copilot) + cache/audit
│   ├── domains/           # Domain services (tabular/, nlp/, data/)
│   ├── observability/     # Prometheus metrics (25), bootstrap, usage, FinOps costs, alerts, reports
│   ├── plugins/           # Domain Plugin Registry (ENABLED_PLUGINS)
│   ├── worker/tasks/      # Celery async tasks (train, predict, train_nlp, predict_nlp)
│   ├── utils/             # Shared utilities
│   ├── database.py        # DB engine & session
│   └── main.py            # FastAPI app factory
├── tests/
│   ├── unit/              # Unit tests
│   └── integration/       # Integration tests
├── migrations/            # Alembic DB migrations
└── pyproject.toml         # Dependencies & project config
```

## Common Tasks

```bash
# Create a new Alembic migration
uv run alembic revision --autogenerate -m "description"

# Apply migrations
uv run alembic upgrade head

# Rollback one migration
uv run alembic downgrade -1

# Format/lint code
uv run ruff check app/ --fix

# Run a specific test
uv run pytest tests/unit/test_preprocessing.py -v -k "test_build_pipeline"
```
