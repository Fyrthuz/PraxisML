# PraxisML — ML Engineering & AI Platform

> **PraxisML** — Fernando González Salas  
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
- **Model Registry (MLflow)**: Registro y versionado de modelos con stages (Staging → Production → Archived). Gestión de versiones y promoción de modelos.
- **Data Registry (DVC)**: Versionado de datasets con DVC. Sincronización automática con MinIO remoto (bucket `praxisml-dvc`). Historial de versiones, descarga de datasets y promoción a producción.
- **Almacenamiento global con MinIO**: Tanto MLflow (artefactos) como DVC (datasets) están configurados para usar **MinIO** como backend de objetos S3-compatible por defecto, facilitando el despliegue en la nube.
- **Observabilidad**: Prometheus + Grafana para monitorización en tiempo real.
- **Frontend React (Next.js)**: Dashboard con polling en tiempo real, drag & drop y gestión visual de recursos.
- **Explicabilidad (XAI)**: Panel de explicabilidad SHAP integrado en resultados de predicción para interpretación de modelos.
- **Streaming en Tiempo Real**: Inferencia continua via WebSockets con cuotas de conexión y manejo de reconexión.
- **Data Drift Monitoring**: Monitorización de estabilidad de distribuciones con umbrales configurables.
- **NLP Engine (Fase 1)**: Entrenamiento e inferencia de modelos transformer (DistilBERT, BERT, RoBERTa, XLM-R, BioBERT) con export **ONNX**, NER con `seqeval`, y análisis de sentimiento.
- **Semantic Search (pgvector)**: Embeddings con `sentence-transformers` (MiniLM-L6-v2, 384 dims) almacenados en PostgreSQL + pgvector para búsqueda semántica sobre modelos y datasets.
- **Asistente LLM + RAG**: Chatbot con gateway multi-provider (**vLLM** default, OpenAI/Ollama/Anthropic), RAG sobre metadatos internos y respuestas en streaming (SSE).
- **NLP Profiler**: Detección automática de columnas de texto, idioma y sugerencia de pipeline al subir datasets.

---

## 📚 Documentación Técnica Detallada

Para comprender a fondo la arquitectura, el diseño de la API y las decisiones técnicas de infraestructura, hemos preparado una extensa documentación técnica en la carpeta `/docs`. **Asegúrate de revisar estos documentos si eres desarrollador o arquitecto de este sistema**:

- 🏗️ [**Arquitectura Global (`docs/architecture.md`)**](docs/architecture.md): Diagrama estandarizado del flujo de vida, componentes y diseño del aislamiento *Multi-Tenant*.
- ⚙️ [**Backend & Motor de ML (`docs/backend.md`)**](docs/backend.md): Estructura detallada del proyecto FastAPI, inyección de dependencias (RBAC), flujos críticos de la API para modelado (TorchScript / Scikit-learn), y cómo opera el control de procesamiento de Celery.
- 🎨 [**Frontend (`docs/frontend.md`)**](docs/frontend.md): Guía de la SPA en React + Next.js. Cómo gestionamos eventos asíncronos (Polling) para evitar congelaciones UI y unificar notificaciones de error.
- 🚀 [**Despliegue y DevOps (`docs/deployment.md`)**](docs/deployment.md): Qué hace cada contenedor en el *Docker compose*, precedencia de las variables de entorno, y cómo inspeccionar logs y telemetría en vivo vía **Grafana** + **Prometheus**.
- 🧠 [**Plan Fase 1 — NLP Engine (`docs/plan-fase-1-nlp-engine.md`)**](docs/plan-fase-1-nlp-engine.md): Especificación completa de la vertical NLP (training, inferencia, embeddings/pgvector, LLM+RAG, profiler) con decisiones de diseño, precios vLLM y observabilidad.

---

## Project Structure

```
PraxisML/
├── backend/                    # FastAPI + Celery backend
│   ├── app/
│   │   ├── api/routes/v1/      # REST endpoints
│   │   │   ├── auth.py         # JWT Login / Register (primer usuario = admin)
│   │   │   ├── tenants.py      # CRUD de tenants + gestión de quotas + budget
│   │   │   ├── datasets.py     # Upload de datasets (ZIP + config.json) + DVC
│   │   │   ├── models.py       # Upload de modelos + MLflow Registry
│   │   │   ├── profiling.py    # Profiling de datasets
│   │   │   ├── drift.py        # Monitorización de data drift
│   │   │   ├── users.py        # Creación/eliminación de usuarios
│   │   │   ├── observability/  # Observability API (Fase 0)
│   │   │   │   └── observability.py  # Usage, costs, pricing, alerts, health
│   │   │   ├── nlp/            # NLP Engine (Fase 1)
│   │   │   │   ├── training.py       # Entrenamiento de transformers (BERT/RoBERTa)
│   │   │   │   ├── inference.py      # Predict, batch y streaming WS NLP
│   │   │   │   ├── embeddings.py     # Embeddings y búsqueda semántica (pgvector)
│   │   │   │   └── assistant.py      # Chatbot LLM + RAG (vLLM/OpenAI/Ollama)
│   │   │   └── tabular/        # Tabular ML domain (renombrado desde root)
│   │   │       ├── training.py       # Entrenamiento Sklearn / PyTorch
│   │   │       ├── predictions.py    # Inferencia async y single-image + XAI
│   │   │       ├── preprocessing.py  # Pipelines de preprocesamiento
│   │   │       └── streaming.py      # Inferencia en tiempo real via WebSockets
│   │   ├── core/               # Seguridad / Config / JWT / Rate Limiting
│   │   │   ├── config.py       # Configuración centralizada con validadores
│   │   │   ├── exceptions.py   # Jerarquía de excepciones de dominio
│   │   │   ├── rate_limit.py   # Instancia slowapi Limiter
│   │   │   └── security.py     # JWT + bcrypt + JWKS
│   │   ├── core_ml/            # Motor de ML e inferencia
│   │   │   ├── factory.py      # PredictionFactory — selecciona estimador
│   │   │   ├── hyperparams.py  # Registro de algoritmos y defaults
│   │   │   ├── preprocessing.py # Pipelines sklearn ColumnTransformer
│   │   │   ├── explainability.py # Explicabilidad SHAP (KernelExplainer)
│   │   │   ├── dataset_parser.py # Parser de datasets ZIP
│   │   │   ├── tabular_parser.py # Parser para datos tabulares (CSV/JSON)
│   │   │   ├── nlp/               # NLP clásico (Fase 1)
│   │   │   │   └── onnx_export.py # Export ONNX con torch.onnx.export
│   │   │   └── uncertainty/    # Estimadores de incertidumbre
│   │   ├── models/             # Modelos SQLAlchemy ORM
│   │   │   ├── embedding.py    # ModelEmbedding + VectorType (pgvector, Fase 1)
│   │   │   └── assistant.py    # AssistantConversation (historial chatbot, Fase 1)
│   │   ├── schemas/            # Schemas Pydantic (request/response)
│   │   │   ├── observability.py  # Schemas de observabilidad (Fase 0)
│   │   │   └── pricing.py      # Schemas de pricing dinámico (Fase 0)
│   │   ├── observability/      # Paquete de observabilidad (Fase 0)
│   │   │   ├── models.py       # ORM: UsageMetric, CostRecord, DriftHistory, AlertConfig, ResourcePricing
│   │   │   ├── metrics.py      # 19 métricas Prometheus custom (incl. NLP/LLM)
│   │   │   ├── usage_collector.py  # Captura métricas de uso
│   │   │   ├── cost_calculator.py  # Estimación de costes con precios dinámicos
│   │   │   ├── alert_engine.py # Motor de evaluación y despacho de alertas
│   │   │   └── reports.py      # Generación de reportes PDF/CSV
│   │   ├── services/           # Lógica de negocio (MLflow, storage, training, DVC)
│   │   │   ├── mlflow_service.py      # MLflow + Model Registry
│   │   │   ├── dvc_service.py         # DVC data versioning
│   │   │   ├── training_service.py    # Entrenamiento de modelos
│   │   │   ├── drift_service.py       # Cálculo de drift con Evidently
│   │   │   ├── data_profiler.py       # Profiling con ydata-profiling
│   │   │   ├── nlp_trainer.py         # Entrenamiento NLP (transformers + ONNX, Fase 1)
│   │   │   ├── nlp_inference.py       # Inferencia NLP (ONNX Runtime, Fase 1)
│   │   │   ├── embedding_service.py   # Embeddings + semantic search (Fase 1)
│   │   │   ├── llm_gateway.py         # Gateway LLM multi-provider (Fase 1)
│   │   │   ├── rag_service.py         # RAG sobre metadatos internos (Fase 1)
│   │   │   ├── nlp_profiler.py        # Detección de columnas de texto (Fase 1)
│   │   │   └── storage_service.py     # Factory de almacenamiento (Local/MinIO/S3)
│   │   ├── worker/             # Workers Celery
│   │   │   └── tasks/          # Tareas async (predict, train, train_nlp, predict_nlp, scheduled_checks)
│   │   ├── utils/              # Utilidades
│   │   │   └── dvc_helper.py   # Helper DVC para CLI
│   │   ├── database.py
│   │   └── main.py
│   ├── migrations/             # Alembic migrations
│   │   └── versions/           # Versiones de schema
│   ├── tests/
│   │   ├── unit/               # Tests unitarios (RBAC, quota, config, ML, registry, observability)
│   │   │   ├── test_observability_models.py
│   │   │   ├── test_observability_metrics.py
│   │   │   ├── test_usage_collector.py
│   │   │   ├── test_cost_calculator.py
│   │   │   ├── test_alert_engine.py
│   │   │   ├── test_observability_reports.py
│   │   │   ├── test_observability_hooks.py
│   │   │   ├── test_services.py
│   │   │   ├── test_nlp_hyperparams.py   # Registry NLP (Fase 1)
│   │   │   ├── test_nlp_trainer.py       # Trainer transformers mockeado (Fase 1)
│   │   │   ├── test_nlp_inference.py     # Inferencia NLP (Fase 1)
│   │   │   ├── test_embedding_service.py # Embeddings + search sqlite (Fase 1)
│   │   │   ├── test_vector_type.py       # VectorType pgvector/JSON (Fase 1)
│   │   │   ├── test_llm_gateway.py       # Gateway LLM mockeado (Fase 1)
│   │   │   ├── test_rag_service.py       # RAG contexto (Fase 1)
│   │   │   └── test_nlp_profiler.py      # NLP Profiler (Fase 1)
│   │   └── integration/        # Tests de integración (API endpoints)
│   │       ├── test_api_endpoints.py
│   │       ├── test_model_lifecycle.py
│   │       ├── test_observability_api.py
│   │       ├── test_tabular_domain.py
│   │       ├── test_deprecated_routes.py
│   │       ├── test_nlp_training_api.py      # API entrenamiento NLP (Fase 1)
│   │       ├── test_nlp_inference_api.py     # API inferencia + WS NLP (Fase 1)
│   │       ├── test_nlp_embeddings_api.py    # API embeddings/search (Fase 1)
│   │       ├── test_nlp_assistant_api.py     # API asistente + SSE (Fase 1)
│   │       └── test_nlp_profiler_hook.py     # Hook profiler en datasets (Fase 1)
│   ├── scripts/                # Scripts utilitarios
│   │   └── validate_model.py   # Validación de modelos para CI/CD
│   └── pyproject.toml
├── frontend/                   # Next.js frontend
│   └── frontend/
│       ├── src/app/            # App Router (Login, Dashboards)
│       ├── src/components/     # UI Components (ExplainabilityPanel, DriftPanel)
│       ├── src/hooks/          # Custom Hooks (useDrift, useStreamingInference)
│       └── src/lib/            # API Client (api.ts) y utilidades
├── infra/                      # Configuración de infraestructura
│   ├── prometheus/             # Scraping de métricas
│   └── grafana/                # Dashboards y provisioning
├── .dvc/                       # Configuración DVC
│   ├── config                  # Config DVC (MinIO/S3)
│   └── config.local           # Overrides locales
├── .github/workflows/          # CI/CD pipelines
│   ├── ci.yml                 # CI principal
│   └── model_ci.yml           # Validación y promoción de modelos
├── .env.example                # Template de variables de entorno
└── docker-compose.yml          # Manifiesto de contenedores
```

---

## Configuration

Toda la configuración se gestiona mediante variables de entorno. El archivo `backend/app/core/config.py` usa **Pydantic Settings** con validación integrada.

### Orden de precedencia (mayor → menor)

1.  **Variables de entorno del sistema** (Docker, CI, shell export)
2.  **Archivo `.env`** (definido por `PRAXISML_ENV_FILE` o CWD)
3.  **Defaults en `config.py`** (solo válidos para desarrollo local)

### Variables principales

| Variable | Default | Valida | Descripción |
|---|---|---|---|
| `ENVIRONMENT` | `development` | `{development, staging, production, testing}` | Entorno — controla seguridad y logging |
| `DATABASE_URL` | `postgresql://...localhost...` | Debe empezar con `postgresql://` | Conexión PostgreSQL |
| `SECRET_KEY` | *(default inseguro)* | **Bloqueado en production** si no se cambia | Clave JWT — generar con `python -c "import secrets; print(secrets.token_urlsafe(64))"` |
| `STORAGE_BACKEND` | `local` | `{local, minio, s3}` | Backend de almacenamiento |
| `PRAXISML_ENV_FILE` | `.env` | Ruta a archivo `.env` | Path personalizado al archivo de entorno |
| `DVC_REMOTE_NAME` | `minio` | Cualquier string | Nombre del remote DVC para dataset versioning |
| `RATE_LIMIT_TRAINING` | `10/minute` | `N/{second,minute,hour,day}` | Rate limit para endpoints de entrenamiento |
| `RATE_LIMIT_INFERENCE` | `30/minute` | `N/{second,minute,hour,day}` | Rate limit para endpoints de inferencia |
| `RATE_LIMIT_LLM` | `20/minute` | `N/{second,minute,hour,day}` | Rate limit para el chatbot LLM (Fase 1) |
| `RATE_LIMIT_NLP_INFERENCE` | `30/minute` | `N/{second,minute,hour,day}` | Rate limit para inferencia NLP (Fase 1) |
| `LLM_PROVIDER` | `vllm` | `{vllm, openai, anthropic, ollama}` | Proveedor LLM del asistente (Fase 1) |
| `VLLM_BASE_URL` | `http://vllm:8001/v1` | URL | Endpoint OpenAI-compatible de vLLM (Fase 1) |
| `VLLM_MODEL_NAME` | `Qwen/Qwen2.5-7B-Instruct` | string | Modelo servido por vLLM (debe coincidir con el pricing) (Fase 1) |
| `LLM_API_KEY` | `EMPTY` | string | API key (vLLM usa `EMPTY`; OpenAI/Anthropic la real) (Fase 1) |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | string | Modelo de embeddings para semantic search (Fase 1) |
| `LLM_CHARGE_COMPUTE` | `false` | `true/false` | Cobrar GPU-hours por request LLM (Fase 1) |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | URL completa | URL de la API para el frontend (WebSocket se convierte automáticamente a ws://) |

> **⚠️ Seguridad en producción**: Con `ENVIRONMENT=production` o `staging`, la app **no arrancará** si la `SECRET_KEY` mantiene el valor por defecto.

#### Configuración del Frontend

Para desplegar el frontend en diferentes entornos, configurar la variable de entorno:

```bash
# .env (frontend)
NEXT_PUBLIC_API_URL=http://localhost:8000
```

Esta variable se usa automáticamente para:
- Construir URLs de API HTTP
- Convertir URLs HTTP a WebSocket (ws://) para streaming

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

### NLP (HuggingFace Transformers — Fase 1)

| Arquitectura | API value | Modelo HF | Notas |
|--------------|-----------|-----------|-------|
| **DistilBERT** | `distilbert` | `distilbert-base-uncased` | Default (rápido, CPU-friendly) |
| BERT Base | `bert-base` | `bert-base-uncased` | |
| BERT Multilingual | `bert-multilingual` | `bert-base-multilingual-cased` | Multiidioma |
| RoBERTa | `roberta` | `roberta-base` | |
| XLM-RoBERTa | `xlm-roberta` | `xlm-roberta-base` | Multilingüe avanzado |
| BioBERT | `biobert` | `dmis-lab/biobert-v1.1` | Dominio médico |

**Tareas soportadas:** `text_classification`, `token_classification` (NER), `sentiment_analysis`.
Los modelos se exportan a **ONNX** (`torch.onnx.export`) para inferencia rápida en CPU.

### LLM Gateway (Fase 1)

| Proveedor | API value | Requiere |
|-----------|-----------|----------|
| **vLLM** (default) | `vllm` | `VLLM_BASE_URL` + `VLLM_MODEL_NAME` (GPU) |
| OpenAI | `openai` | `LLM_API_KEY` |
| Ollama | `ollama` | `OLLAMA_BASE_URL` |
| Anthropic | `anthropic` | `LLM_API_KEY` |

---

## Model Registry (MLflow)

Sistema de gestión de versiones de modelos con stages:

| Stage | Descripción |
|-------|-------------|
| **Staging** | Modelo en pruebas/validación (default) |
| **Production** | Modelo en producción activo |
| **Archived** | Modelo archivado (no disponible para inferencia) |

### Flujo de uso

1.  **Gestionar versiones**: Inspeccionar cada versión para ver métricas, parámetros y tags asociados. Promover a Production o archivar.
2.  **Descargar modelos**: Botón de descarga directa en cada versión del registry para obtener un ZIP con el modelo, pipeline y metadatos.
3.  **CI/CD**: Workflow automático (`model_ci.yml`) para validar métricas antes de promoción.
4.  **Multi-tenant isolation**: Los nombres en el Registry se prefijan automáticamente con `tenant_{id}_` para garantizar aislamiento y visibilidad correcta en la UI.

### Campos del modelo

| Campo | Descripción |
|-------|-------------|
| `version` | Versión semántica (1.0.0, 1.0.1, etc.) |
| `stage` | Stage actual en MLflow Registry |
| `promoted_at` | Fecha de última promoción |
| `promoted_by` | Usuario que promovió el modelo |
| `mlflow_registry_name` | Nombre en MLflow Registry |
| `mlflow_version` | Versión en MLflow |

---

## Data Registry (DVC)

Sistema de versionado de datasets con sincronización a MinIO/S3:

### Características

-   **Versionado automático**: Cada upload crea una nueva versión
-   **Sincronización remote**: Push/pull automático a MinIO/S3
-   **Hashing**: MD5 hash para integridad de datos
-   **Promoción**: Marcar datasets como "Production"

### Flujo de uso

1.  **Subir con DVC**: Al subir un dataset, activar "Track with DVC"
2.  **Ver versiones**: Ir al tab "Data Registry" para ver el historial
3.  **Gestionar**: Promover a producción, push/pull desde remote

### Campos del dataset

| Campo | Descripción |
|-------|-------------|
| `dvc_hash` | Hash MD5 del archivo |
| `is_dvc_tracked` | Si está trackeado con DVC |
| `dvc_registry_name` | Nombre del registry |
| `dvc_version` | Versión en DVC |
| `dvc_remote` | Remote configurado (minio/s3) |

---

## 🏗️ Arquitectura de API (Fase 0.6)

### Rutas Tabulares Reorganizadas

La API tabular ha sido reorganizada bajo el prefijo `/api/v1/tabular/` para preparar la estructura multi-dominio (NLP, CV, Time Series). Las rutas antiguas siguen funcionando como aliases de compatibilidad.

| Dominio | Prefijo | Endpoints |
|---------|---------|-----------|
| **Tabular Training** | `/api/v1/tabular/training/` | `GET /algorithms`, `POST /train`, `GET /status/{task_id}` |
| **Tabular Predictions** | `/api/v1/tabular/` | `POST /predict`, `POST /predictions/predict/single`, `GET /predictions` |
| **Tabular Preprocessing** | `/api/v1/tabular/preprocessing/` | `POST /preview`, `POST /apply/{dataset_id}`, `GET /pipeline/{dataset_id}` |
| **Tabular Streaming** | `/api/v1/tabular/` | `WS /streaming/predict/{model_id}` |

> **Compatibilidad**: Las rutas antiguas (`/api/v1/training/`, `/api/v1/predictions/`, etc.) siguen funcionando pero están ocultas del OpenAPI y marcadas como deprecated.

---

## 📊 Observability & Cost Tracking (Fase 0)

Sistema unificado de observabilidad multi-tenant para monitorización de uso, costes y alertas.

### Características

- **Usage Tracking**: Registro automático de cada inferencia y entrenamiento con latencia, dominio, y tags
- **Dynamic Pricing**: Precios dinámicos para GPU, CPU, storage y tokens LLM (CRUD via API)
- **Cost Calculator**: Estimación automática de costes con cache Redis (TTL 5min)
- **Budget Management**: Budget mensual por tenant con alertas de overrun
- **Alert System**: 15 alertas predefinidas (cost spike, latency degradation, drift, etc.)
- **Health Score**: Score compuesto (accuracy + drift + latency) por modelo
- **Reports**: Generación de reportes mensuales en PDF/CSV

### API Endpoints

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `GET` | `/api/v1/observability/usage` | Métricas de uso agregadas |
| `GET` | `/api/v1/observability/costs` | Desglose de costes por tipo |
| `GET` | `/api/v1/observability/costs/forecast` | Predicción de coste mensual |
| `GET` | `/api/v1/observability/pricing` | Precios activos (GPU, CPU, storage, LLM) |
| `POST` | `/api/v1/observability/pricing` | Crear/actualizar precio (admin) |
| `GET` | `/api/v1/observability/alerts` | Alertas configuradas |
| `POST` | `/api/v1/observability/alerts` | Crear nueva alerta |
| `GET` | `/api/v1/observability/models/{id}/health` | Health score de un modelo |
| `GET` | `/api/v1/observability/drift/history/{model_id}` | Historial de drift |
| `GET` | `/api/v1/observability/tenant/{id}/report` | Reporte mensual PDF/CSV |

### Frontend Dashboard

El tab **Observability** en el frontend muestra:
- **Overview**: 4 stat cards (coste mensual, métricas de uso, alertas activas, forecast)
- **Usage**: Tabla de métricas atómicas con stats agregadas (avg, p50, p95, p99, min, max)
- **Costs**: Tabla detallada de cost records por tipo y modelo
- **Alerts**: CRUD completo de alertas con toggle on/off
- **Pricing**: Tabla de precios agrupada por categoría (GPU, CPU, storage, LLM tokens)

### Métricas Prometheus

19 métricas custom expuestas en `/metrics`:
- `praxisml_inference_latency_ms` (Histogram)
- `praxisml_inference_tokens_total` (Counter)
- `praxisml_inference_cost_usd` (Counter)
- `praxisml_training_duration_seconds` (Histogram)
- `praxisml_training_cost_usd` (Counter)
- `praxisml_model_storage_bytes` (Gauge)
- `praxisml_ws_connections_active` (Gauge)
- `praxisml_drift_score` (Gauge)
- `praxisml_quota_usage_pct` (Gauge)
- `praxisml_cost_budget_remaining` (Gauge)
- `praxisml_predictions_total` (Counter)
- `praxisml_errors_total` (Counter)
- `praxisml_nlp_inference_latency_ms` (Histogram, Fase 1)
- `praxisml_nlp_predictions_total` (Counter, Fase 1)
- `praxisml_embeddings_total` (Counter, Fase 1)
- `praxisml_semantic_search_latency_ms` (Histogram, Fase 1)
- `praxisml_llm_tokens_total` (Counter, Fase 1)
- `praxisml_llm_requests_total` (Counter, Fase 1)
- `praxisml_llm_latency_ms` (Histogram, Fase 1)

### Grafana Dashboards

6 dashboards nuevos + 5 existentes enriquecidos:
- **Cost Center**: Costes por tenant, breakdown por modelo, forecast vs budget
- **Model Performance**: Latencia p50/p95/p99, throughput, error rate, drift score
- **Tenant Usage**: Requests/día, entrenamientos, almacenamiento, cuotas
- **LLM Spend**: Tokens input/output, coste por modelo, budget burn rate
- **Training Dashboard**: Duración por algoritmo, coste por entrenamiento, GPU utilization
- **Drift Timeline**: PSI/KS timeline, forecast line, alert annotations
- **NLP Engine** (Fase 1): Entrenamiento NLP (p95), latencia inferencia, predicciones, embeddings, latencia búsqueda semántica, tokens/latencia/requests LLM

---

## 🧠 NLP Engine (Fase 1)

Vertical completa de NLP bajo el router `/api/v1/nlp/*` con dos ejes: **NLP Clásico** (transformers fine-tuneados) y **NLP Generativo** (LLM + RAG).

### Componentes

| Componente | Descripción |
|------------|-------------|
| **NLP Training** | Fine-tuning de BERT/RoBERTa/DistilBERT sobre CSVs (`text_column` + `label_column`). Tareas: clasificación, NER, sentimiento. Métricas NER con `seqeval`. Autologging en MLflow (`tenant_{id}_nlp_training`). |
| **ONNX Export** | Los modelos se exportan a ONNX (`torch.onnx.export`) para inferencia rápida en CPU con `onnxruntime`. |
| **NLP Inference** | `POST /predict` (síncrono), `POST /predict/batch` (Celery sobre CSV) y `WS /streaming/{model_id}` (tiempo real). Devuelve `{label, confidence, entities[], tokens[], latency_ms}`. |
| **Semantic Search** | Embeddings `all-MiniLM-L6-v2` (384 dims) en `model_embeddings` (pgvector, índice IVFFlat). Búsqueda coseno sobre modelos y datasets del tenant. |
| **LLM Gateway** | Abstracción multi-provider (vLLM default). Registra tokens y coste por tenant (`CostRecord` cost_type=`llm`). |
| **Asistente RAG** | Chatbot que construye contexto desde metadatos internos (modelos, drift, costes, búsqueda semántica). **Nunca recibe datos crudos de los datasets.** Streaming SSE. |
| **NLP Profiler** | Post-hook en `POST /datasets/`: detecta columnas de texto, idioma (langdetect), vocabulario estimado y sugiere pipeline. Se guarda en `dataset.column_types_analysis`. |

### Base de datos (migración `phase1_nlp_001`)

- Extensión `vector` (pgvector) + tabla `model_embeddings(id, tenant_id, model_id, dataset_id, embedding vector(384), metadata jsonb)`.
- Tabla `assistant_conversations(id, tenant_id, user_id, messages jsonb)`.
- Columna `dataset.column_types_analysis JSON` (salida del profiler).
- Seed de precios vLLM en `resource_pricing` (`llm_tokens` → precio 0.0 self-hosted, administrable vía `POST /observability/pricing`).

### Frontend

El tab **NLP** del dashboard incluye 4 sub-tabs:
- **Training**: formulario dataset + columnas + arquitectura + hiperparámetros + estado del job.
- **Inference**: predicción de texto con label/confianza/entidades/latencia + batch CSV.
- **Semantic Search**: búsqueda semántica sobre modelos/datasets con score.
- **Assistant**: chat con streaming SSE, fuentes citadas y borrado de historial.

> 📖 Plan de implementación completo en [`docs/plan-fase-1-nlp-engine.md`](docs/plan-fase-1-nlp-engine.md).

---

## ⚡ Quick Start (Docker)

### Prerrequisitos
-   [Docker Desktop](https://www.docker.com/products/docker-desktop/)
-   Node.js ≥ 18 (para desarrollo del frontend)

### 1. Configurar entorno
```bash
cp .env.example .env
# Editar .env con tus valores (SECRET_KEY es crítico para producción)
```

### 2. Levantar todos los servicios (infraestructura + API + worker)
```bash
docker compose up --build
```

Para levantar solo infraestructura (sin API/worker, para desarrollo local):
```bash
docker compose up -d db redis mlflow minio minio-init
```

### 3. Levantar frontend (desarrollo)
```bash
cd frontend/frontend
npm i
npm run dev
```

### 3b. (Opcional) Levantar el LLM (vLLM) para el asistente RAG
Requiere GPU. Añade el servicio `vllm` y arranca con el perfil `nlp`:
```bash
docker compose --profile nlp up -d vllm
# Configura en .env: LLM_PROVIDER=vllm, VLLM_BASE_URL=http://localhost:8001/v1
```

### 4. Servicios disponibles
| Servicio | URL | Descripción |
|----------|-----|-------------|
| **API (FastAPI)** | `http://localhost:8000` | Backend REST |
| **Swagger UI** | `http://localhost:8000/docs` | Documentación interactiva |
| **Celery Worker** | *(interno)* | Procesamiento asíncrono |
| **MLflow UI** | `http://localhost:5000` | Experiment tracking |
| **MLflow UI local** | `http://localhost:5001` | Alternativa (solo dev) |
| **MinIO Console** | `http://localhost:9001` | Almacenamiento objetos |
| **PostgreSQL (pgvector)** | `localhost:5432` | Base de datos + vector store |
| **Redis** | `localhost:6379` | Cache + message broker |
| **vLLM** *(opcional)* | `http://localhost:8001` | Servidor LLM OpenAI-compatible |
| **Prometheus** | `http://localhost:9090` | Métricas |
| **Grafana** | `http://localhost:3001` (admin/admin) | Dashboards |

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
docker compose up -d db redis mlflow minio minio-init
```

### 3. Arrancar backend
```bash
cd backend
uv sync
uv run alembic upgrade head
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
uv run pytest tests/ -v --cov=app --cov-fail-under=50
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

### Usuarios
| Método | Path | Rol | Descripción |
|--------|------|-----|-------------|
| GET | `/api/v1/users/` | admin | Listar todos los usuarios del tenant |
| POST | `/api/v1/users/` | admin | Crear nuevo usuario en el tenant |
| GET | `/api/v1/users/{id}` | admin | Obtener detalles de un usuario |
| PATCH | `/api/v1/users/{id}/role` | admin | Cambiar el rol de un usuario |
| DELETE | `/api/v1/users/{id}` | admin | Desactivar un usuario (soft delete) |

### Datasets
| Método | Path | Rol | Descripción |
|--------|------|-----|-------------|
| POST | `/api/v1/datasets/` | editor | Subir dataset (`.zip` + `config.json`) + opcional DVC |
| GET | `/api/v1/datasets/` | viewer | Listar datasets del tenant |
| DELETE | `/api/v1/datasets/{id}` | admin | Eliminar dataset |
| GET | `/api/v1/datasets/registry` | viewer | Listar registries DVC |
| GET | `/api/v1/datasets/registry/{name}/versions` | viewer | Ver versiones de un dataset |
| POST | `/api/v1/datasets/{id}/promote` | editor | Promover dataset a producción |
| POST | `/api/v1/datasets/{id}/dvc/push` | editor | Subir a remote DVC |
| POST | `/api/v1/datasets/{id}/dvc/pull` | editor | Descargar de remote DVC |

### Modelos
| Método | Path | Rol | Descripción |
|--------|------|-----|-------------|
| POST | `/api/v1/models/upload` | editor | Subir modelo TorchScript (`.pt`) |
| POST | `/api/v1/models/` | editor | Registrar modelo desde MLflow run |
| GET | `/api/v1/models/` | viewer | Listar modelos (tenant + públicos) |
| DELETE | `/api/v1/models/{id}` | admin | Eliminar modelo y datos asociados |
| POST | `/api/v1/models/{id}/promote` | editor | Promover modelo a Production/Archived |
| POST | `/api/v1/models/{id}/archive` | editor | Archivar modelo |
| GET | `/api/v1/models/{id}/versions` | viewer | Ver historial de versiones |

### Model Registry (MLflow)
| Método | Path | Rol | Descripción |
|--------|------|-----|-------------|
| POST | `/api/v1/models/registry` | editor | Crear registered model |
| GET | `/api/v1/models/registry` | viewer | Listar registered models |
| DELETE | `/api/v1/models/registry/{name}` | admin | Eliminar registered model |

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
| POST | `/api/v1/predictions/predict/single` | editor | Inferencia directa sobre una imagen o tabular |
| GET | `/api/v1/predictions/status/{id}` | viewer | Consultar estado de la predicción |
| GET | `/api/v1/predictions` | viewer | Historial de predicciones |
| GET | `/api/v1/predictions/{id}/explain` | viewer | Obtener valores SHAP para una predicción (explicabilidad) |
| POST | `/api/v1/streaming/predict/{model_id}` | editor | **Streaming WebSocket**: Inferencia en tiempo real (conecta via WebSocket) |

### Preprocesamiento
| Método | Path | Rol | Descripción |
|--------|------|-----|-------------|
| POST | `/api/v1/preprocessing/preview` | editor | Previsualizar transformaciones |
| POST | `/api/v1/preprocessing/apply` | editor | Aplicar pipeline y guardar en MLflow |
| GET | `/api/v1/preprocessing/pipelines` | viewer | Consultar pipelines guardados |

### Data Drift
| Método | Path | Rol | Descripción |
|--------|------|-----|-------------|
| GET | `/api/v1/drift/report/{id}` | viewer | Obtener reporte de drift |
| PATCH | `/api/v1/drift/thresholds/{id}` | editor | Actualizar umbrales (PSI/KS) |

### Profiling
| Método | Path | Rol | Descripción |
|--------|------|-----|-------------|
| GET | `/api/v1/profiling/{dataset_id}` | viewer | Obtener profiling de un dataset |

### NLP Training (Fase 1)
| Método | Path | Rol | Descripción |
|--------|------|-----|-------------|
| GET | `/api/v1/nlp/algorithms` | viewer | Listar arquitecturas NLP disponibles |
| POST | `/api/v1/nlp/train` | editor | Lanzar entrenamiento NLP (Celery, rate-limited) |
| GET | `/api/v1/nlp/training/status/{task_id}` | viewer | Estado del entrenamiento NLP |

### NLP Inference (Fase 1)
| Método | Path | Rol | Descripción |
|--------|------|-----|-------------|
| POST | `/api/v1/nlp/predict` | editor | Clasificar texto / extraer entidades (síncrono) |
| POST | `/api/v1/nlp/predict/batch` | editor | Batch inference sobre CSV (asíncrono, 202) |
| GET | `/api/v1/nlp/predict/batch/{id}/result` | viewer | Resultados JSON del batch |
| WS | `/api/v1/nlp/streaming/{model_id}` | editor | Clasificación NLP en tiempo real |

### Semantic Search (Fase 1)
| Método | Path | Rol | Descripción |
|--------|------|-----|-------------|
| POST | `/api/v1/nlp/embed` | viewer | Generar embeddings de texto libre |
| GET | `/api/v1/nlp/search` | viewer | Búsqueda semántica sobre modelos/datasets |

### Asistente LLM (Fase 1)
| Método | Path | Rol | Descripción |
|--------|------|-----|-------------|
| POST | `/api/v1/nlp/assistant/chat` | editor | Chat con el asistente (JSON o SSE streaming) |
| GET | `/api/v1/nlp/assistant/history` | viewer | Historial de conversaciones |
| DELETE | `/api/v1/nlp/assistant/history` | editor | Borrar historial del asistente |

---

## CI Pipeline

Automatizado con GitHub Actions (`.github/workflows/`):

### CI Principal (`ci.yml`)

| Paso | Comando |
|------|---------|
| **Lint** | `ruff check app/ --select=E,F,W --ignore=E501` |
| **Tests unitarios** | `pytest tests/unit/ -v` |
| **Tests de integración** | `pytest tests/integration/ -v` |
| **Cobertura** | `pytest tests/ --cov=app --cov-fail-under=50` |

### Model CI (`model_ci.yml`)

Workflow para validación y promoción automática de modelos:

| Paso | Descripción |
|------|-------------|
| **Validación** | Compara métricas del modelo con thresholds definidos |
| **Promoción** | Si pasa validación, promueve automáticamente a Production |
| **Notificación** | Comenta en el PR/Issue con el resultado |

Se ejecuta manualmente con:
```bash
gh workflow run model_ci.yml -f run_id=<MLFLOW_RUN_ID> -f target_stage=Production
```

---

## Tech Stack

| Capa | Tecnología |
|------|-----------|
| Framework API | FastAPI |
| Cola asíncrona | Celery + Redis |
| ML & Deep Learning | Scikit-learn, PyTorch, TorchScript |
| **NLP Clásico** | **HuggingFace Transformers + ONNX Runtime + seqeval** |
| **Embeddings / Vector Store** | **sentence-transformers + pgvector** |
| **LLM Gateway / RAG** | **vLLM (OpenAI-compatible) / OpenAI / Ollama / Anthropic** |
| Autenticación | OAuth2 JWT + bcrypt |
| Autorización | RBAC (admin/editor/viewer) + quotas |
| Rate Limiting | slowapi (por IP) |
| Experiment Tracking | MLflow |
| Model Registry | MLflow Registry |
| Data Versioning | DVC |
| Almacenamiento objetos | MinIO (S3-compatible) / AWS S3 |
| Base de datos | PostgreSQL + pgvector + SQLAlchemy + Alembic |
| Migraciones | Alembic |
| Monitorización | Prometheus + Grafana |
| Gestión de paquetes | uv + pyproject.toml |
| Frontend | React (Next.js) + Tailwind + react-hot-toast |
| CI/CD | GitHub Actions |
