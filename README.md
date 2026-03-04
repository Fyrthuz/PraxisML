# Antigravity SaaS — Medical Image Segmentation & Uncertainty Estimation

> **TFM Productivo** — Fernando González Salas  
> Productivization of research in Bayesian uncertainty estimation for medical image segmentation.

---

## Overview

This project provides a production-ready API backend for running uncertainty-aware deep learning inference on medical images (MRI segmentation). It wraps research algorithms (MC Dropout, TTA, Noisy Inference and their **Ensemble**) in a secure, multi-tenant SaaS architecture featuring:

- **JWT Hybrid Authentication & Multi-Tenancy**: Secure REST API isolated by organizational tenants, supporting both local secure JWT minting (bcrypt + python-jose) and external providers.
- **Continuous Safe Model execution**: Dynamic ingestion of PyTorch `TorchScript (.pt)` files ensuring secure server-side execution without arbitrary Python injections.
- **FastAPI** REST API
- **Celery + Redis** for async heavy inference workloads
- **Declarative Datasets & Single Image uploads**: Bypass heavy datasets for quick inferences or leverage robust JSON configuration-driven dataset uploads.
- **MLFlow** for full model and inference lifecycle tracking
- **PostgreSQL** for data persistence
- **Next.js Frontend (React)**: Incorporating real-time Celery task polling, reactive error propagation (`react-hot-toast`), and drag-and-drop file inference.

---

## Project Structure

```
TFM_productivo/
├── backend/                    # FastAPI + Celery backend
│   ├── app/
│   │   ├── api/routes/v1/      # REST endpoints
│   │   │   ├── auth.py         # JWT Login / Register
│   │   │   ├── tenants.py
│   │   │   ├── datasets.py     # Configurable Zip + JSON handling
│   │   │   ├── models.py       # TorchScript upload + MLFlow auto-registration
│   │   │   └── predictions.py  # Async prediction & single-image inference
│   │   ├── core/               # Security / Config / JWT dependencies
│   │   ├── core_ml/            # ML inference engine
│   │   │   ├── dataset_parser.py # Declarative PyTorch config parsing
│   │   │   ├── factory.py      # PredictionFactory — selects estimator by name
│   │   │   └── uncertainty/
│   │   │       ├── mc_dropout.py
│   │   │       ├── tta.py
│   │   │       ├── noise_inference.py
│   │   │       └── ensemble.py
│   │   ├── models/             # SQLAlchemy ORM models (Tenant, User, etc.)
│   │   ├── worker/
│   │   │   ├── celery_app.py
│   │   │   └── tasks/
│   │   │       └── predict.py  # Dynamic TorchScript load + MLFlow track
│   │   ├── database.py
│   │   └── main.py
│   └── pyproject.toml
├── frontend/                   # Next.js frontend
│   └── frontend/
│       ├── src/app/            # App Router (Login, Dashboards)
│       └── src/components/     # AuthProvider, Modals, Drag&Drop Zones
├── models/                     # Sample models
├── MRI/                        # Sample data
└── docker-compose.yml          # Unified container manifest
```

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

## ⚡ Quick Start (Integration with Docker)

The easiest way to run the entire backend stack (PostgreSQL, Redis, MLFlow, API, and Worker) is using **Docker Compose**.

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) 
- Node.js ≥ 18 (For running the frontend)

### 1. Launch Backend Services
Run this in the project root:
```bash
docker-compose up --build
```
> Note: If you encounter an `mlflow_service.py` mounting error, ensure Docker file sharing is enabled for the project directory.

### 2. Launch Next.js Frontend
```bash
cd frontend/frontend
npm i
npm run dev
```

### 3. Available Services
| Service | URL |
|---|---|
| **Frontend** | `http://localhost:3000` |
| **API (FastAPI)** | `http://localhost:8000/docs` |
| **MLFlow UI** | `http://localhost:5000` |

---

## 🛠️ Manual Development Setup

If you prefer to run services individually for faster code iterations:

### 1. Configure backend
Create `backend/app/.env` (or set environment variables):
```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/antigravity_db
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
MLFLOW_TRACKING_URI=http://localhost:5000
SECRET_KEY=your_super_secret_jwt_symmetric_key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440
```

### 2. Start backing services (Docker)
We use Docker for the required services:
```bash
docker-compose up -d db redis mlflow
```

### 3. Run migrations & start backend API
```bash
cd backend
uv sync
.venv\Scripts\python.exe migrate.py
.venv\Scripts\uvicorn.exe app.main:app --reload
```

### 4. Start Celery worker
```bash
cd backend
.venv\Scripts\celery.exe -A app.worker.celery_app worker --loglevel=info --pool=solo
```

---

## Core API Endpoints

Interactive docs available at **`http://localhost:8000/docs`**

### Authentication & Tenants
| Method | Path | Description |
|---|---|---|
| POST | `/api/v1/auth/register` | Register user and create root Tenant scope |
| POST | `/api/v1/auth/login` | Obtain OAuth2 JWT Bearer Token |
| GET | `/api/v1/auth/me` | Decode token for active profile |

### Datasets
| Method | Path | Description |
|---|---|---|
| POST | `/api/v1/datasets/` | Upload `.zip` datasets accompanied by a declarative `config.json` |
| GET | `/api/v1/datasets/` | List JWT-scoped datasets |

### Models
| Method | Path | Description |
|---|---|---|
| POST | `/api/v1/models/upload` | **Upload Safe `.pt` (TorchScript)** |
| GET | `/api/v1/models/` | List models intersecting `<tenant_id>` OR `is_public=True` |

### Predictions
| Method | Path | Description |
|---|---|---|
| POST | `/api/v1/predictions/predict` | Dispatch standard inference to Celery queue |
| POST | `/api/v1/predictions/predict/single` | **Bypass Datasets**: Direct raw image injection for on-the-fly inference. |
| GET | `/api/v1/predictions/status/{id}` | Long-poll task state, error propagation natively streams back JSON failures |
| GET | `/api/v1/predictions` | List inference history |

---

## MLFlow Tracking & Dynamic Loading

With security as a core architectural principle, `.py` class inference has been abstracted away:

1. **Model registration**: Developers utilize `torch.jit.trace` locally to produce robust static artifacts. `POST /api/v1/models/upload` registers these within MLFlow while flipping `is_torchscript=True`.
2. **Dynamic Spawning**: The Celery engine dynamically invokes `torch.jit.load(map_location=device)` natively within isolated threads.
3. Every inference traces down hyper-parameters, metric benchmarks (like `inference_time_s` and `max_uncertainty`), and exports `.npy` variance mappings cleanly.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Application Framework | FastAPI |
| Async Queue | Celery + Redis |
| Deep Learning backend | PyTorch & TorchScript |
| Authentication | OAuth2 JWT & bcrypt hashing |
| Experiment tracking | MLFlow |
| Relational Schema | PostgreSQL + SQLAlchemy |
| Package management | uv + pyproject.toml |
| Frontend | React (Next.js) + Tailwind + react-hot-toast |
