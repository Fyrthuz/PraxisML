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
# Frontend:  http://localhost:3000
# API docs:  http://localhost:8000/docs
# MLflow:    http://localhost:5000
# MinIO:     http://localhost:9001
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

Generate a secure SECRET_KEY:

```bash
python -c "import secrets; print(secrets.token_urlsafe(64))"
```

## Project Structure

```
backend/
├── app/
│   ├── api/routes/v1/     # REST API endpoints (auth, datasets, models, training, etc.)
│   ├── core/              # Config, security, logging, middleware
│   ├── core_ml/           # ML engine (preprocessing, uncertainty, explainability)
│   ├── models/            # SQLAlchemy ORM models
│   ├── schemas/           # Pydantic request/response schemas
│   ├── services/          # Business logic (MLflow, DVC, training, inference)
│   ├── worker/tasks/      # Celery async tasks
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
