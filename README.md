# PraxisML — Medical Image Segmentation & Uncertainty Estimation

> **TFM Productivo** — Fernando González Salas  
> Productivization of research in Bayesian uncertainty estimation for medical image segmentation.

---

## Overview

This project provides a production-ready API backend for running uncertainty-aware deep learning inference on medical images (MRI segmentation). It wraps research algorithms (MC Dropout, TTA, Noisy Inference and their **Ensemble**) in a secure, multi-tenant SaaS architecture featuring:

- **JWT Hybrid Authentication & Multi-Tenancy**: Secure REST API isolated by organizational tenants, supporting both local secure JWT minting (bcrypt + python-jose) and external providers.
- **Role-Based Access Control (RBAC)**: Three-tier role hierarchy (`admin` → `editor` → `viewer`) enforced per endpoint.
- **Quota & Rate Limiting**: Configurable per-tenant resource quotas and per-IP rate limiting via `slowapi`.
- **Continuous Safe Model execution**: Dynamic ingestion of PyTorch `TorchScript (.pt)` files ensuring secure server-side execution without arbitrary Python injections.
- **FastAPI** REST API with structured logging and Prometheus metrics.
- **Celery + Redis** for async heavy inference workloads.
- **Declarative Datasets & Single Image uploads**: Bypass heavy datasets for quick inferences or leverage robust JSON configuration-driven dataset uploads.
- **MLFlow** for full model and inference lifecycle tracking.
- **PostgreSQL** for data persistence with **MinIO** (S3-compatible) object storage.
- **Next.js Frontend (React)**: Incorporating real-time Celery task polling, reactive error propagation (`react-hot-toast`), and drag-and-drop file inference.
- **Observability**: Prometheus + Grafana dashboards for monitoring.

---

## Project Structure

```
TFM_productivo/
├── backend/                    # FastAPI + Celery backend
│   ├── app/
│   │   ├── api/routes/v1/      # REST endpoints
│   │   │   ├── auth.py         # JWT Login / Register (first user = admin)
│   │   │   ├── tenants.py      # Tenant CRUD + quota management
│   │   │   ├── datasets.py     # Configurable Zip + JSON handling
│   │   │   ├── models.py       # TorchScript upload + MLFlow auto-registration
│   │   │   ├── predictions.py  # Async prediction & single-image inference
│   │   │   ├── training.py     # Sklearn/PyTorch training pipelines
│   │   │   ├── preprocessing.py # Data preprocessing pipelines
│   │   │   └── profiling.py    # Dataset profiling
│   │   ├── core/               # Security / Config / JWT / Rate Limiting
│   │   │   ├── config.py       # Centralized config with validators
│   │   │   ├── exceptions.py   # Custom exception hierarchy
│   │   │   ├── rate_limit.py   # slowapi Limiter instance
│   │   │   └── security.py     # JWT + bcrypt + JWKS
│   │   ├── core_ml/            # ML inference engine
│   │   │   ├── factory.py      # PredictionFactory — selects estimator by name
│   │   │   ├── hyperparams.py  # Algorithm registry & defaults
│   │   │   ├── preprocessing.py # sklearn ColumnTransformer pipelines
│   │   │   └── uncertainty/
│   │   │       ├── mc_dropout.py
│   │   │       ├── tta.py
│   │   │       ├── noise_inference.py
│   │   │       ├── ensemble.py
│   │   │       └── sklearn_uncertainty.py
│   │   ├── models/             # SQLAlchemy ORM models
│   │   ├── schemas/            # Pydantic request/response schemas
│   │   ├── services/           # Business logic (MLFlow, storage, training)
│   │   ├── worker/
│   │   │   ├── celery_app.py
│   │   │   └── tasks/
│   │   │       ├── predict.py
│   │   │       ├── single_predict.py
│   │   │       └── train.py
│   │   ├── database.py
│   │   └── main.py
│   ├── tests/
│   │   ├── unit/               # Unit tests (RBAC, quota, ML core)
│   │   └── integration/        # Integration tests (API endpoints)
│   ├── migrate.py              # Database migrations
│   └── pyproject.toml
├── frontend/                   # Next.js frontend
│   └── frontend/
│       ├── src/app/            # App Router (Login, Dashboards)
│       └── src/components/     # AuthProvider, Modals, Drag&Drop Zones
├── infra/                      # Infrastructure configs
│   ├── prometheus/
│   └── grafana/
├── .env.example                # Template for environment variables
└── docker-compose.yml          # Unified container manifest
```

---

## Configuration

All settings are managed through environment variables. The configuration file at `backend/app/core/config.py` uses **Pydantic Settings** with built-in validation.

### Precedence Order (highest → lowest)

1. **System environment variables** (Docker, CI, shell export)
2. **`.env` file** (project root or CWD)
3. **Defaults in `config.py`** (only suitable for local development)

### Key Variables

| Variable | Default | Validates | Description |
|---|---|---|---|
| `ENVIRONMENT` | `development` | `{development, staging, production, testing}` | App environment — controls security, logging |
| `DATABASE_URL` | `postgresql://...localhost...` | Must start with `postgresql://` | PostgreSQL connection string |
| `SECRET_KEY` | *(insecure default)* | **Blocked in production** if unchanged | JWT signing key — generate with `python -c "import secrets; print(secrets.token_urlsafe(64))"` |
| `CORS_ORIGINS` | `http://localhost:3000,...` | Comma-separated URLs | Allowed CORS origins |
| `STORAGE_BACKEND` | `local` | `{local, minio, s3}` | File storage backend |
| `RATE_LIMIT_TRAINING` | `10/minute` | `N/{second,minute,hour,day}` | Rate limit for training endpoints |
| `RATE_LIMIT_INFERENCE` | `30/minute` | `N/{second,minute,hour,day}` | Rate limit for prediction endpoints |

> **⚠️ Production Safety**: Setting `ENVIRONMENT=production` or `staging` with the default `SECRET_KEY` will **crash on startup** with a clear error message — preventing accidental deployment with insecure credentials.

See [`.env.example`](.env.example) for the full list of configurable variables.

---

## Security: RBAC & Quotas

### Role-Based Access Control

| Role | Level | Capabilities |
|------|-------|-------------|
| `admin` | 3 | Full access: create tenants, delete resources, manage quotas |
| `editor` | 2 | Create/upload datasets & models, run training & predictions |
| `viewer` | 1 | Read-only: list, preview, download, view status |

The first user registered in a tenant is automatically assigned the `admin` role.

### Tenant Quotas

| Quota | Default | Description |
|-------|---------|-------------|
| `max_datasets` | 50 | Maximum datasets per tenant |
| `max_models` | 20 | Maximum models per tenant |
| `max_predictions_per_day` | 500 | Daily prediction limit |
| `max_training_jobs_per_day` | 10 | Daily training job limit |

Quotas are configurable per tenant via the `PATCH /api/v1/tenants/{id}/quotas` endpoint (admin only). Setting a quota to `null` means unlimited.

---

## Uncertainty Methods

| Method | API value | Description |
|---|---|---|
| MC Dropout | `mc_dropout` | Stochastic dropout at inference time. Entropy of averaged predictions. |
| TTA | `tta` | Random affine augmentations, inverted back to original space. Entropy of mean. |
| Noisy Inference | `noisy_inference` | Gaussian noise perturbations on input. Entropy of mean probabilities. |
| **Ensemble** | `ensemble` | **Combines all three** (weighted avg. probs + epistemic variance). |
| None | `none` | Standard inference without uncertainty. Returns zero entropy map. |

### Ensemble Strategy

The `EnsembleUncertaintyEstimator` runs all three methods in parallel on the same model and fuses their outputs:

```
Prediction  = w_mc · P_mc + w_tta · P_tta + w_noise · P_noise
Aleatoric   = w_mc · H_mc + w_tta · H_tta + w_noise · H_noise
Epistemic   = Var([P_mc, P_tta, P_noise]).mean(classes)
Uncertainty = (1 - α) · Aleatoric + α · Epistemic
```

Default weights are equal (`[1,1,1]`) and `α = 0.5`. Can be overridden via API.

---

## ⚡ Quick Start (Docker)

The easiest way to run the entire stack is using **Docker Compose**.

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- Node.js ≥ 18 (for frontend development)

### 1. Configure environment
```bash
cp .env.example .env
# Edit .env with your values (SECRET_KEY is critical for production)
```

### 2. Launch all services
```bash
docker-compose up --build
```

### 3. Launch Next.js Frontend (development)
```bash
cd frontend/frontend
npm i
npm run dev
```

### 4. Available Services
| Service | URL |
|---|---|
| **Frontend** | `http://localhost:3000` |
| **API (FastAPI)** | `http://localhost:8000/docs` |
| **MLFlow UI** | `http://localhost:5000` |
| **MinIO Console** | `http://localhost:9001` |
| **Prometheus** | `http://localhost:9090` |
| **Grafana** | `http://localhost:3001` (admin/admin) |

---

## 🛠️ Manual Development Setup

If you prefer to run services individually for faster code iterations:

### 1. Configure environment
```bash
cp .env.example .env
# Edit values for local development (localhost URLs instead of Docker hostnames)
```

### 2. Start backing services (Docker)
```bash
docker-compose up -d db redis mlflow minio minio-init
```

### 3. Run migrations & start backend API
```bash
cd backend
uv sync
uv run python migrate.py
uv run uvicorn app.main:app --reload
```

### 4. Start Celery worker
```bash
cd backend
uv run celery -A app.worker.celery_app worker --loglevel=info --pool=solo
```

### 5. Run tests
```bash
cd backend
uv run pytest tests/ -v --cov=app --cov-fail-under=30
```

---

## Core API Endpoints

Interactive docs available at **`http://localhost:8000/docs`**

### Authentication & Tenants
| Method | Path | Role | Description |
|---|---|---|---|
| POST | `/api/v1/auth/register` | — | Register user and create root Tenant scope |
| POST | `/api/v1/auth/login` | — | Obtain OAuth2 JWT Bearer Token |
| GET | `/api/v1/auth/me` | any | Decode token for active profile |
| POST | `/api/v1/tenants/` | admin | Create new tenant |
| PATCH | `/api/v1/tenants/{id}/quotas` | admin | Update tenant quotas |

### Datasets
| Method | Path | Role | Description |
|---|---|---|---|
| POST | `/api/v1/datasets/` | editor | Upload `.zip` datasets with a declarative `config.json` |
| GET | `/api/v1/datasets/` | viewer | List JWT-scoped datasets |
| DELETE | `/api/v1/datasets/{id}` | admin | Delete a dataset |

### Models
| Method | Path | Role | Description |
|---|---|---|---|
| POST | `/api/v1/models/upload` | editor | **Upload Safe `.pt` (TorchScript)** |
| POST | `/api/v1/models/` | editor | Register model from MLFlow run |
| GET | `/api/v1/models/` | viewer | List models (tenant + public) |
| DELETE | `/api/v1/models/{id}` | admin | Delete model and associated data |

### Training
| Method | Path | Role | Description |
|---|---|---|---|
| POST | `/api/v1/training/train` | editor | Launch sklearn/PyTorch training (rate-limited) |
| GET | `/api/v1/training/algorithms` | viewer | List available algorithms |
| GET | `/api/v1/training/status/{task_id}` | viewer | Poll training task state |

### Predictions
| Method | Path | Role | Description |
|---|---|---|---|
| POST | `/api/v1/predictions/predict` | editor | Dispatch inference to Celery queue (rate-limited) |
| POST | `/api/v1/predictions/predict/single` | editor | Direct image injection for on-the-fly inference |
| GET | `/api/v1/predictions/status/{id}` | viewer | Poll task state with JSON error propagation |
| GET | `/api/v1/predictions` | viewer | List inference history |

---

## MLFlow Tracking & Dynamic Loading

With security as a core architectural principle, `.py` class inference has been abstracted away:

1. **Model registration**: Developers utilize `torch.jit.trace` locally to produce robust static artifacts. `POST /api/v1/models/upload` registers these within MLFlow while flipping `is_torchscript=True`.
2. **Dynamic Spawning**: The Celery engine dynamically invokes `torch.jit.load(map_location=device)` natively within isolated threads.
3. Every inference traces down hyper-parameters, metric benchmarks (like `inference_time_s` and `max_uncertainty`), and exports `.npy` variance mappings cleanly.

---

## CI Pipeline

Automated via GitHub Actions (`.github/workflows/ci.yml`):

| Step | Command |
|---|---|
| **Lint** | `ruff check app/ --select=E,F,W --ignore=E501` |
| **Unit tests** | `pytest tests/unit/ -v` |
| **Integration tests** | `pytest tests/integration/ -v` |
| **Coverage** | `pytest tests/ --cov=app --cov-fail-under=30` |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Application Framework | FastAPI |
| Async Queue | Celery + Redis |
| Deep Learning backend | PyTorch & TorchScript, Scikit-learn |
| Authentication | OAuth2 JWT & bcrypt hashing |
| Authorization | RBAC (admin/editor/viewer) + quota limiting |
| Rate Limiting | slowapi (per-IP) |
| Experiment tracking | MLFlow |
| Object storage | MinIO (S3-compatible) |
| Relational Schema | PostgreSQL + SQLAlchemy |
| Monitoring | Prometheus + Grafana |
| Package management | uv + pyproject.toml |
| Frontend | React (Next.js) + Tailwind + react-hot-toast |
| CI/CD | GitHub Actions |
