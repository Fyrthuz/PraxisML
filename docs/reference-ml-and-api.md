# Referencia Técnica: Algoritmos ML/NLP, Seguridad, Registries y API

## 1. Algoritmos de ML y NLP Soportados

### Scikit-learn (Tabular)

| Algoritmo | API value | Clasificación | Regresión |
|-----------|-----------|:---:|:---:|
| Random Forest | `random_forest` | ✅ | ✅ |
| Gradient Boosting | `gradient_boosting` | ✅ | ✅ |
| SVM | `svm` | ✅ | ✅ |
| KNN | `knn` | ✅ | ✅ |
| Logistic / Linear Regression | `logistic_regression` | ✅ | ✅ |
| Decision Tree | `decision_tree` | ✅ | ✅ |
| AdaBoost | `adaboost` | ✅ | ✅ |

### PyTorch (Tabular)

| Arquitectura | API value | Descripción |
|-------------|-----------|-------------|
| MLP | `mlp` | Perceptrón multicapa configurable (capas, dropout, activación) |
| Custom | *(registrado)* | Cualquier `nn.Module` registrado en `ModelFactory` |

### Métodos de Incertidumbre

| Método | API value | Descripción |
|--------|-----------|-------------|
| MC Dropout | `mc_dropout` | Dropout estocástico en inferencia |
| TTA | `tta` | Augmentaciones aleatorias invertidas |
| Noisy Inference | `noisy_inference` | Perturbaciones gaussianas en la entrada |
| Ensemble | `ensemble` | Combina los métodos anteriores |
| Predictive Entropy | `entropy` | Entropía predictiva sobre probabilidades |
| Tree Variance | `tree_variance` | Varianza entre árboles en ensembles |
| Conformal Prediction | `conformal` | Intervalos/conjuntos conformales calibrados |
| Ninguno | `none` | Inferencia estándar sin incertidumbre |

### NLP (HuggingFace Transformers + ONNX)

| Arquitectura | API value | Modelo Base HF | Notas |
|--------------|-----------|----------------|-------|
| **DistilBERT** | `distilbert` | `distilbert-base-uncased` | Default (rápido, CPU-friendly) |
| BERT Base | `bert-base` | `bert-base-uncased` | |
| BERT Multilingual | `bert-multilingual` | `bert-base-multilingual-cased` | Multiidioma |
| RoBERTa | `roberta` | `roberta-base` | |
| XLM-RoBERTa | `xlm-roberta` | `xlm-roberta-base` | Multilingüe avanzado |
| BioBERT | `biobert` | `dmis-lab/biobert-v1.1` | Dominio biomédico |

- **Tareas soportadas**: `text_classification`, `token_classification` (NER con `seqeval`), `sentiment_analysis`.
- **Export ONNX**: nativo con `torch.onnx.export` (`dynamo=False`) para inferencia rápida en CPU vía `onnxruntime`.

### LLM Gateway (Asistente RAG)

| Proveedor | API value | Configuración requerida |
|-----------|-----------|-------------------------|
| **vLLM** (default) | `vllm` | `VLLM_BASE_URL` + `VLLM_MODEL_NAME` (OpenAI-compatible) |
| OpenAI | `openai` | `LLM_API_KEY` |
| Ollama | `ollama` | `OLLAMA_BASE_URL` |
| Anthropic | `anthropic` | `LLM_API_KEY` |

Si el proveedor LLM externo no está disponible, el asistente realiza fallback automático a síntesis RAG local sobre los metadatos del tenant (sin exponer nunca filas crudas).

---

## 2. Explicabilidad (XAI)

Implementada en `backend/app/core_ml/explainability.py`:

- **Tabular (`explain_tabular_fast`)**:
  1. `shap.TreeExplainer` para modelos basados en árboles (exacto en pocos milisegundos, incluso sobre 1 sola fila).
  2. Atribución lineal (`coef_ * x`) para modelos lineales.
  3. Perturbación por feature (`f(x) - f(x_i=0)`) como fallback universal para PyTorch/SVM/KNN.
- **NLP (`explain_nlp_text`)**: atribución por palabra mediante oclusión (`token_occlusion`), midiendo la variación de confianza al omitir cada palabra con el runtime ONNX/Transformers cargado.
- Disponible mediante `explain: true` en:
  - `POST /serving/v1/{slug}/invocations` (Tabular y NLP)
  - `POST /api/v1/nlp/predict`
  - `POST /api/v1/tabular/predict`, `/predictions/predict/single`, `/predictions/predict/batch` (persistido como `shap.json` y consultable en `GET /api/v1/tabular/predictions/{id}/explain`)
  - `WS /api/v1/tabular/streaming/predict/{model_id}`

---

## 3. Seguridad: RBAC, Quotas y Aislamiento Multi-Tenant

### Roles RBAC

| Rol | Nivel | Permisos |
|-----|-------|----------|
| `admin` | 3 | Acceso total: crear tenants, gestionar usuarios, quotas, pricing y eliminar recursos |
| `editor` | 2 | Subir datasets y modelos, crear pipelines/entornos/endpoints, entrenar y ejecutar inferencia |
| `viewer` | 1 | Solo lectura: listar, previsualizar, descargar y consultar métricas/logs |

El primer usuario registrado en un tenant recibe automáticamente el rol `admin`.

### Quotas por Tenant

| Quota | Default | Descripción |
|-------|---------|-------------|
| `max_datasets` | `50` | Máximo de datasets por tenant |
| `max_models` | `20` | Máximo de modelos por tenant |
| `max_predictions_per_day` | `500` | Límite diario de predicciones |
| `max_training_jobs_per_day` | `10` | Límite diario de entrenamientos |

Configurables vía `PATCH /api/v1/tenants/{id}/quotas` (`admin`). Valor `null` = sin límite.

### Seguridad Enterprise
- **PostgreSQL Row-Level Security (RLS)**: contexto inyectado por transacción (`SET LOCAL app.current_tenant_id`).
- **API Keys (`px_...`)**: almacenadas con hash SHA-256 y scopes granulares (`serving:invoke`, etc.).
- **Guardias de Compatibilidad**: bloqueo en UI y API (`HTTP 400`) ante cruces inválidos entre modelos Tabular y NLP en comparador, Canary pairing e inferencia.

---

## 4. Model Registry (MLflow) & Data Registry (DVC)

### MLflow Tracking & Model Registry
- **Experimentos jerárquicos (`MLFlowService.ensure_tenant_experiment`)**: `{tenant_slug}/{domain}` (o `tenant-{short_id}/{domain}`) para `tabular-training`, `nlp-training`, `preprocessing`, `pipelines`, `models` e `inference`, etiquetados con `praxisml.tenant_id`, `praxisml.domain` y `praxisml.managed_by=praxisml`.
- **Stages del Registry**: `Staging` (validación) → `Production` (activo) → `Archived` (retirado).
- **Campos principales del modelo**: `version`, `stage`, `promoted_at`, `promoted_by`, `mlflow_registry_name`, `mlflow_version`.

### DVC Data Registry & MinIO (`StorageKeyBuilder`)
- **Claves jerárquicas en bucket `praxisml`**:
  - `tenants/{tenant_id}/datasets/raw/{filename}`
  - `tenants/{tenant_id}/datasets/processed/{filename}`
  - `tenants/{tenant_id}/datasets/pipelines/{filename}`
  - `tenants/{tenant_id}/datasets/configs/{filename}`
  - `tenants/{tenant_id}/artifacts/models/{model_id}/{filename}`
  - `tenants/{tenant_id}/predictions/{prediction_id}/{filename}`
- **Versionado DVC** en bucket dedicado `praxisml-dvc` con campos `dvc_hash` (MD5), `is_dvc_tracked`, `dvc_registry_name`, `dvc_version` y `dvc_remote`.

---

## 5. Referencia Completa de API Endpoints

Documentación interactiva OpenAPI en `http://localhost:8000/docs`.

### Auth, Tenants, Users & Governance

| Método | Path | Rol | Descripción |
|--------|------|-----|-------------|
| `POST` | `/api/v1/auth/register` | — | Registrar usuario + crear tenant |
| `POST` | `/api/v1/auth/login` | — | Obtener JWT Bearer Token |
| `GET` | `/api/v1/auth/me` | any | Perfil del usuario autenticado |
| `GET / POST` | `/api/v1/tenants/` | viewer / admin | Listar o crear tenants |
| `PATCH` | `/api/v1/tenants/{id}/quotas` | admin | Actualizar quotas del tenant |
| `GET / POST / PATCH / DELETE` | `/api/v1/users/*` | admin | Gestión de usuarios y roles RBAC |
| `GET / POST / DELETE` | `/api/v1/api-keys/*` | editor / admin | Personal Access Tokens (`px_...`) |
| `GET / POST` | `/api/v1/workspaces/*` | viewer / admin | Gestión de workspaces (`dev`/`staging`/`prod`) |
| `GET` | `/api/v1/audit/` & `/audit/events` | viewer | Audit Log y eventos Redis Streams |

### Datasets, DVC, Profiling, Drift, Environments & Pipelines

| Método | Path | Rol | Descripción |
|--------|------|-----|-------------|
| `POST / GET / DELETE` | `/api/v1/datasets/*` | editor / viewer / admin | Ingesta, listado, preview y borrado |
| `GET / POST` | `/api/v1/datasets/registry/*` | viewer / editor | Versiones DVC, promoción y push/pull |
| `GET` | `/api/v1/profiling/{dataset_id}` | viewer | Perfilado estadístico y detección de texto |
| `GET / PATCH` | `/api/v1/drift/{report,thresholds}/{id}` | viewer / editor | Análisis de drift (PSI/KS) y umbrales |
| `GET` | `/api/v1/environments/capabilities` | viewer | Inspección de los 7 registros pluggables |
| `GET / POST` | `/api/v1/environments/` | viewer / editor | Gestión de `RuntimeEnvironments` (`uv`/`docker`) |
| `GET / POST` | `/api/v1/pipelines/` & `/{id}/execute` | viewer / editor | `UserPipelines` (`declarative`/`python`/`hybrid`) |

### Tabular ML & NLP Engine

| Método | Path | Rol | Descripción |
|--------|------|-----|-------------|
| `POST` | `/api/v1/tabular/preprocessing/{preview,apply/{dataset_id}}` | editor | Previsualización, renombrado libre y guardado en MLflow |
| `GET / POST` | `/api/v1/tabular/training/{algorithms,train,status/{task_id}}` | viewer / editor | Entrenamiento Sklearn / PyTorch |
| `POST / GET` | `/api/v1/tabular/predict` · `/predictions/*` | editor / viewer | Inferencia batch/single, datos y `/{id}/explain` (SHAP) |
| `WS` | `/api/v1/tabular/streaming/predict/{model_id}` | editor | Inferencia tabular en tiempo real + SHAP |
| `GET / POST` | `/api/v1/nlp/{algorithms,train,training/status/{task_id}}` | viewer / editor | Fine-tuning HuggingFace + ONNX |
| `POST / GET / WS` | `/api/v1/nlp/predict` · `/predict/batch` · `/streaming/{id}` | editor / viewer | Inferencia NLP síncrona (`explain`), batch y WS |
| `POST / GET` | `/api/v1/nlp/{embed,search}` | viewer | Embeddings y búsqueda semántica (`pgvector`) |
| `POST / GET / DELETE` | `/api/v1/nlp/assistant/{chat,history}` | editor / viewer | Chat RAG en JSON o SSE streaming |

### Models, Registry, Serving, Observability & Copilot

| Método | Path | Rol | Descripción |
|--------|------|-----|-------------|
| `GET / POST / DELETE` | `/api/v1/models/*` · `/models/registry/*` | viewer / editor / admin | Modelos, versiones, promoción, archivo y descarga |
| `GET / POST / PATCH / DELETE` | `/api/v1/serving/endpoints/*` | viewer / editor | Endpoints dedicados, Canary split, `/metrics` y `/logs` |
| `POST` | `/serving/v1/{slug}/invocations` | viewer / API Key | Invocación productiva Tabular o NLP (`explain: true`) |
| `GET / POST` | `/api/v1/observability/*` | viewer / editor / admin | `platform-summary`, `usage`, `costs`, `costs/forecast`, `pricing`, `alerts`, `models/{id}/health`, `tenant/{id}/report` |
| `GET` | `/api/v1/copilot/{plugins,tools}` | viewer | Catálogo de plugins activos y herramientas PolicyEngine |
