# PraxisML — ML Engineering & AI Platform

> **TFM Productivo** — Fernando González Salas  
> Plataforma integral para el desarrollo, entrenamiento, despliegue y monitorización de modelos de Machine Learning e Inteligencia Artificial.

---

## Overview

**PraxisML** es una plataforma SaaS multi-tenant para el ciclo completo de ML Engineering: desde la ingesta y preprocesamiento de datos, pasando por el entrenamiento y evaluación de modelos (Scikit-learn, PyTorch), hasta la inferencia en producción con estimación de incertidumbre. Diseñada para equipos que necesitan trazabilidad, seguridad y escalabilidad desde el día uno.

### Características principales

- **Multi-Tenancy & Autenticación JWT**: API REST aislada por organizaciones, con soporte para JWT local (bcrypt + python-jose) y proveedores externos (Auth0, Clerk).
- **RBAC (Control de Acceso por Roles)**: Jerarquía de tres niveles (`admin` → `editor` → `viewer`) con enforcement por endpoint.
- **Quotas & Rate Limiting**: Límites de recursos configurables por tenant y rate limiting por IP vía `slowapi`.
- **Entrenamiento de modelos**: Pipelines completos para Scikit-learn y PyTorch con holdout y cross-validation, hiperparámetros configurables y autologging en MLflow.
- **Preprocesamiento configurable**: Pipeline declarativo (JSON) soportando escalado, imputación, encoding, feature engineering y selección de columnas.
- **Inferencia con incertidumbre**: MC Dropout, TTA, Noisy Inference y Ensemble — aplicables a cualquier modelo PyTorch.
- **Ejecución segura de modelos**: Ingesta dinámica de TorchScript (`.pt`) sin ejecución arbitraria de Python.
- **Procesamiento asíncrono**: Celery + Redis para tareas pesadas (entrenamiento, inferencia batch).
- **Experiment Tracking**: MLflow para trazabilidad completa de experimentos, métricas, artefactos y pipelines.
- **Almacenamiento flexible**: Local, MinIO (S3-compatible) o AWS S3.
- **Observabilidad**: Prometheus + Grafana para monitorización en tiempo real.
- **Frontend React (Next.js)**: Dashboard con polling en tiempo real, drag & drop y gestión visual de recursos.

---

## 📚 Documentación Técnica Detallada

Para comprender a fondo la arquitectura, el diseño de la API y las decisiones técnicas de infraestructura, hemos preparado una extensa documentación técnica en la carpeta `/docs`. **Asegúrate de revisar estos documentos si eres desarrollador o arquitecto de este sistema**:

- 🏗️ [**Arquitectura Global (`docs/architecture.md`)**](docs/architecture.md): Diagrama estandarizado del flujo de vida, componentes y diseño del aislamiento *Multi-Tenant*.
- ⚙️ [**Backend & Motor de ML (`docs/backend.md`)**](docs/backend.md): Estructura detallada del proyecto FastAPI, inyección de dependencias (RBAC), flujos críticos de la API para modelado (TorchScript / Scikit-learn), y cómo opera el control de procesamiento de Celery.
- 🎨 [**Frontend (`docs/frontend.md`)**](docs/frontend.md): Guía de la SPA en React + Next.js. Cómo gestionamos eventos asíncronos (Polling) para evitar congelaciones UI y unificar notificaciones de error.
- 🚀 [**Despliegue y DevOps (`docs/deployment.md`)**](docs/deployment.md): Qué hace cada contenedor en el *Docker compose*, precedencia de las variables de entorno, y cómo inspeccionar logs y telemetría en vivo vía **Grafana** + **Prometheus**.

---

## Project Structure

```
TFM_productivo/
├── backend/                    # FastAPI + Celery backend
│   ├── app/
│   │   ├── api/routes/v1/      # REST endpoints
│   │   │   ├── auth.py         # JWT Login / Register (primer usuario = admin)
│   │   │   ├── tenants.py      # CRUD de tenants + gestión de quotas
│   │   │   ├── datasets.py     # Upload de datasets (ZIP + config.json)
│   │   │   ├── models.py       # Upload de modelos + registro en MLflow
│   │   │   ├── predictions.py  # Inferencia async y single-image
│   │   │   ├── training.py     # Entrenamiento Sklearn / PyTorch
│   │   │   ├── preprocessing.py # Pipelines de preprocesamiento
│   │   │   └── profiling.py    # Profiling de datasets
│   │   ├── core/               # Seguridad / Config / JWT / Rate Limiting
│   │   │   ├── config.py       # Configuración centralizada con validadores
│   │   │   ├── exceptions.py   # Jerarquía de excepciones de dominio
│   │   │   ├── rate_limit.py   # Instancia slowapi Limiter
│   │   │   └── security.py     # JWT + bcrypt + JWKS
│   │   ├── core_ml/            # Motor de ML e inferencia
│   │   │   ├── factory.py      # PredictionFactory — selecciona estimador
│   │   │   ├── hyperparams.py  # Registro de algoritmos y defaults
│   │   │   ├── preprocessing.py # Pipelines sklearn ColumnTransformer
│   │   │   └── uncertainty/    # Estimadores de incertidumbre
│   │   ├── models/             # Modelos SQLAlchemy ORM
│   │   ├── schemas/            # Schemas Pydantic (request/response)
│   │   ├── services/           # Lógica de negocio (MLflow, storage, training)
│   │   ├── worker/             # Workers Celery
│   │   │   └── tasks/          # Tareas async (predict, train)
│   │   ├── database.py
│   │   └── main.py
│   ├── tests/
│   │   ├── unit/               # Tests unitarios (RBAC, quota, config, ML)
│   │   └── integration/        # Tests de integración (API endpoints)
│   ├── migrate.py              # Migraciones de base de datos
│   └── pyproject.toml
├── frontend/                   # Next.js frontend
│   └── frontend/
│       ├── src/app/            # App Router (Login, Dashboards)
│       └── src/components/     # AuthProvider, Modals, Drag&Drop
├── infra/                      # Configuración de infraestructura
│   ├── prometheus/             # Scraping de métricas
│   └── grafana/                # Dashboards y provisioning
├── .env.example                # Template de variables de entorno
└── docker-compose.yml          # Manifiesto de contenedores
```

---

## Configuration

Toda la configuración se gestiona mediante variables de entorno. El archivo `backend/app/core/config.py` usa **Pydantic Settings** con validación integrada.

### Orden de precedencia (mayor → menor)

1. **Variables de entorno del sistema** (Docker, CI, shell export)
2. **Archivo `.env`** (raíz del proyecto o CWD)
3. **Defaults en `config.py`** (solo válidos para desarrollo local)

### Variables principales

| Variable | Default | Valida | Descripción |
|---|---|---|---|
| `ENVIRONMENT` | `development` | `{development, staging, production, testing}` | Entorno — controla seguridad y logging |
| `DATABASE_URL` | `postgresql://...localhost...` | Debe empezar con `postgresql://` | Conexión PostgreSQL |
| `SECRET_KEY` | *(default inseguro)* | **Bloqueado en production** si no se cambia | Clave JWT — generar con `python -c "import secrets; print(secrets.token_urlsafe(64))"` |
| `STORAGE_BACKEND` | `local` | `{local, minio, s3}` | Backend de almacenamiento |
| `RATE_LIMIT_TRAINING` | `10/minute` | `N/{second,minute,hour,day}` | Rate limit para endpoints de entrenamiento |
| `RATE_LIMIT_INFERENCE` | `30/minute` | `N/{second,minute,hour,day}` | Rate limit para endpoints de inferencia |

> **⚠️ Seguridad en producción**: Con `ENVIRONMENT=production` o `staging`, la app **no arrancará** si la `SECRET_KEY` mantiene el valor por defecto.

Ver [`.env.example`](.env.example) para la lista completa.

---

## Seguridad: RBAC & Quotas

### Control de Acceso por Roles (RBAC)

| Rol | Nivel | Permisos |
|-----|-------|----------|
| `admin` | 3 | Acceso total: crear tenants, eliminar recursos, gestionar quotas |
| `editor` | 2 | Crear/subir datasets y modelos, lanzar entrenamiento e inferencia |
| `viewer` | 1 | Solo lectura: listar, previsualizar, descargar, consultar estado |

El primer usuario registrado en un tenant recibe automáticamente el rol `admin`.

### Quotas por Tenant

| Quota | Default | Descripción |
|-------|---------|-------------|
| `max_datasets` | 50 | Máximo de datasets por tenant |
| `max_models` | 20 | Máximo de modelos por tenant |
| `max_predictions_per_day` | 500 | Límite diario de predicciones |
| `max_training_jobs_per_day` | 10 | Límite diario de trabajos de entrenamiento |

Configurables vía `PATCH /api/v1/tenants/{id}/quotas` (solo admin). Valor `null` = sin límite.

---

## Algoritmos de ML soportados

### Scikit-learn

| Algoritmo | API value | Clasificación | Regresión |
|-----------|-----------|:---:|:---:|
| Random Forest | `random_forest` | ✅ | ✅ |
| Gradient Boosting | `gradient_boosting` | ✅ | ✅ |
| SVM | `svm` | ✅ | ✅ |
| KNN | `knn` | ✅ | ✅ |
| Logistic / Linear Regression | `logistic_regression` | ✅ | ✅ |
| Decision Tree | `decision_tree` | ✅ | ✅ |
| AdaBoost | `adaboost` | ✅ | ✅ |

### PyTorch

| Arquitectura | API value | Descripción |
|-------------|-----------|-------------|
| MLP | `mlp` | Perceptrón multicapa configurable (capas, dropout, activación) |
| Custom | *(registrado)* | Cualquier `nn.Module` registrado en `ModelFactory` |

### Métodos de incertidumbre

| Método | API value | Descripción |
|--------|-----------|-------------|
| MC Dropout | `mc_dropout` | Dropout estocástico en inferencia |
| TTA | `tta` | Augmentaciones aleatorias invertidas |
| Noisy Inference | `noisy_inference` | Perturbaciones gaussianas en la entrada |
| **Ensemble** | `ensemble` | Combina los tres métodos anteriores |
| Ninguno | `none` | Inferencia estándar sin incertidumbre |

---

## ⚡ Quick Start (Docker)

### Prerrequisitos
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- Node.js ≥ 18 (para desarrollo del frontend)

### 1. Configurar entorno
```bash
cp .env.example .env
# Editar .env con tus valores (SECRET_KEY es crítico para producción)
```

### 2. Levantar todos los servicios
```bash
docker-compose up --build
```

### 3. Levantar frontend (desarrollo)
```bash
cd frontend/frontend
npm i
npm run dev
```

### 4. Servicios disponibles
| Servicio | URL |
|----------|-----|
| **Frontend** | `http://localhost:3000` |
| **API (Swagger)** | `http://localhost:8000/docs` |
| **MLflow UI** | `http://localhost:5000` |
| **MinIO Console** | `http://localhost:9001` |
| **Prometheus** | `http://localhost:9090` |
| **Grafana** | `http://localhost:3001` (admin/admin) |

---

## 🛠️ Desarrollo local

Para iterar más rápido sin reconstruir contenedores:

### 1. Configurar entorno
```bash
cp .env.example .env
# Editar con URLs localhost en vez de hostnames Docker
```

### 2. Levantar servicios de soporte
```bash
docker-compose up -d db redis mlflow minio minio-init
```

### 3. Arrancar backend
```bash
cd backend
uv sync
uv run python migrate.py
uv run uvicorn app.main:app --reload
```

### 4. Arrancar worker Celery
```bash
cd backend
uv run celery -A app.worker.celery_app worker --loglevel=info --pool=solo
```

### 5. Ejecutar tests
```bash
cd backend
uv run pytest tests/ -v --cov=app --cov-fail-under=30
```

---

## API Endpoints

Documentación interactiva en **`http://localhost:8000/docs`**

### Autenticación
| Método | Path | Rol | Descripción |
|--------|------|-----|-------------|
| POST | `/api/v1/auth/register` | — | Registrar usuario + crear tenant |
| POST | `/api/v1/auth/login` | — | Obtener JWT Bearer Token |
| GET | `/api/v1/auth/me` | any | Perfil del usuario autenticado |

### Tenants
| Método | Path | Rol | Descripción |
|--------|------|-----|-------------|
| POST | `/api/v1/tenants/` | admin | Crear nuevo tenant |
| GET | `/api/v1/tenants/` | viewer | Listar tenants |
| PATCH | `/api/v1/tenants/{id}/quotas` | admin | Actualizar quotas del tenant |

### Datasets
| Método | Path | Rol | Descripción |
|--------|------|-----|-------------|
| POST | `/api/v1/datasets/` | editor | Subir dataset (`.zip` + `config.json`) |
| GET | `/api/v1/datasets/` | viewer | Listar datasets del tenant |
| DELETE | `/api/v1/datasets/{id}` | admin | Eliminar dataset |

### Modelos
| Método | Path | Rol | Descripción |
|--------|------|-----|-------------|
| POST | `/api/v1/models/upload` | editor | Subir modelo TorchScript (`.pt`) |
| POST | `/api/v1/models/` | editor | Registrar modelo desde MLflow run |
| GET | `/api/v1/models/` | viewer | Listar modelos (tenant + públicos) |
| DELETE | `/api/v1/models/{id}` | admin | Eliminar modelo y datos asociados |

### Entrenamiento
| Método | Path | Rol | Descripción |
|--------|------|-----|-------------|
| POST | `/api/v1/training/train` | editor | Lanzar entrenamiento (rate-limited) |
| GET | `/api/v1/training/algorithms` | viewer | Listar algoritmos disponibles |
| GET | `/api/v1/training/status/{task_id}` | viewer | Consultar estado del entrenamiento |

### Predicciones
| Método | Path | Rol | Descripción |
|--------|------|-----|-------------|
| POST | `/api/v1/predictions/predict` | editor | Despachar inferencia async (rate-limited) |
| POST | `/api/v1/predictions/predict/single` | editor | Inferencia directa sobre una imagen |
| GET | `/api/v1/predictions/status/{id}` | viewer | Consultar estado de la predicción |
| GET | `/api/v1/predictions` | viewer | Historial de predicciones |

### Preprocesamiento
| Método | Path | Rol | Descripción |
|--------|------|-----|-------------|
| POST | `/api/v1/preprocessing/preview` | editor | Previsualizar transformaciones |
| POST | `/api/v1/preprocessing/apply` | editor | Aplicar pipeline y guardar en MLflow |
| GET | `/api/v1/preprocessing/pipelines` | viewer | Consultar pipelines guardados |

---

## CI Pipeline

Automatizado con GitHub Actions (`.github/workflows/ci.yml`):

| Paso | Comando |
|------|---------|
| **Lint** | `ruff check app/ --select=E,F,W --ignore=E501` |
| **Tests unitarios** | `pytest tests/unit/ -v` |
| **Tests de integración** | `pytest tests/integration/ -v` |
| **Cobertura** | `pytest tests/ --cov=app --cov-fail-under=30` |

---

## Tech Stack

| Capa | Tecnología |
|------|-----------|
| Framework API | FastAPI |
| Cola asíncrona | Celery + Redis |
| ML & Deep Learning | Scikit-learn, PyTorch, TorchScript |
| Autenticación | OAuth2 JWT + bcrypt |
| Autorización | RBAC (admin/editor/viewer) + quotas |
| Rate Limiting | slowapi (por IP) |
| Experiment Tracking | MLflow |
| Almacenamiento objetos | MinIO (S3-compatible) / AWS S3 |
| Base de datos | PostgreSQL + SQLAlchemy |
| Monitorización | Prometheus + Grafana |
| Gestión de paquetes | uv + pyproject.toml |
| Frontend | React (Next.js) + Tailwind + react-hot-toast |
| CI/CD | GitHub Actions |
