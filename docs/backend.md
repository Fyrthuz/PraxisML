# Backend Documentation (FastAPI)

El backend expone una API REST asíncrona robusta y maneja el motor de Machine Learning central. Está escrito en **Python 3.10+** usando `uv` para gestión de paquetes.

## 1. Stack Tecnológico
- **Framework Web**: `FastAPI` (asíncrono, validaciones automáticas, OpenAPI generada).
- **ORM**: `SQLAlchemy 2.0`
- **Gestión de Migraciones**: `Alembic`
- **Cola de Tareas**: `Celery` (Broker: Redis).
- **Control de Versiones y Tracking ML**: `MLflow` backend store.
- **Machine Learning**: `PyTorch`, `TorchScript`, `Scikit-learn`, `Pandas`, `Numpy`.
- **NLP Clásico**: `HuggingFace Transformers`, `onnxruntime` (inferencia ONNX), `seqeval` (métricas NER).
- **Embeddings / Vector Store**: `sentence-transformers` (MiniLM-L6-v2) + PostgreSQL `pgvector`.
- **LLM Gateway**: SDK `openai` (compatible vLLM) + providers `openai`/`anthropic`/`ollama`.
- **Integridad y Validaciones**: `Pydantic v2` / `Pydantic Settings`.
- **RBAC & Auth**: `PyJWT`, `passlib` (bcrypt), `slowapi`.

## 2. Estructura de Directorios Claves (`backend/app/`)

- **/api/routes/v1/**: Rutas REST (`tenants`, `models`, `datasets`, `training`, `predictions`, `preprocessing`, `profiling`, etc.).
- **/api/routes/v1/nlp/**: Vertical NLP (`/api/v1/nlp/*`):
    - `training.py`: `GET /algorithms`, `POST /train`, `GET /training/status/{task_id}`.
    - `inference.py`: `POST /predict`, `POST /predict/batch`, `GET /predict/batch/{id}/result`, `WS /streaming/{model_id}`.
    - `embeddings.py`: `POST /embed`, `GET /search`.
    - `assistant.py`: `POST /assistant/chat` (JSON o SSE), `GET/DELETE /assistant/history`.
- **/api/deps.py**: Inyección de dependencias (Sesiones DB, validaciones de tokens JWT, chequeo de roles RBAC, limitación de cuotas).
- **/core/**: Configuraciones maestras (`config.py`), jerarquía de excepciones (`exceptions.py`), logs estructurados (`logging.py`), motor de limitación (`rate_limit.py`) y seguridad JWT.
- **/core_ml/**: Motor central de predicciones e incertidumbre libre de dependencias HTTP. Define la capa base del inferenciador Scikit-learn y PyTorch, parseadores de datasets declarativos y familias de algoritmos.
    - `/core_ml/nlp/onnx_export.py`: Export ONNX nativo con `torch.onnx.export`.
- **/models/**: Declaraciones SQLAlchemy ORM.
    - `embedding.py`: `ModelEmbedding` + tipo `VectorType` (pgvector en PostgreSQL, JSON en SQLite para tests).
    - `assistant.py`: `AssistantConversation` (historial del chatbot).
- **/schemas/**: Validadores Pydantic de Request y Response.
- **/platform/**: Los 7 registros pluggables y capas transversales:
    - `storage/`: `StorageBackendRegistry` (`local`, `minio`, `s3`) y `StorageKeyBuilder`.
    - `compute/`: `ComputeBackendRegistry` (`celery`) y `EnvironmentRunnerRegistry` (`uv_isolated`, `docker`).
    - `serving/`: `ModelRuntimeRegistry` (`tabular_generic`, `nlp`, `pyfunc`) y motor Canary/A-B.
    - `data_engine/`: `DataEngineRegistry` (`pandas`).
    - `tracker/`: `TrackerRegistry` y `MLFlowService` (experimentos jerárquicos).
    - `events/`, `audit/`, `cache/`, `copilot/`: Redis Streams EventBus, Audit Log, ModelCache y PolicyEngine.
- **/domains/**: Servicios de negocio agrupados por vertical:
    - `tabular/`: `training_service.py`, `training_utils.py`, `inference_service.py`.
    - `nlp/`: `trainer.py`, `inference.py`, `embedding_service.py`, `llm_gateway.py`, `rag_service.py`, `profiler.py`.
    - `data/`: `dvc_service.py`, `data_profiler.py`.
    - `nlp_inference.py`: Inferencia NLP con ONNX Runtime o fallback transformers, integrada con `ModelCache`.
    - `embedding_service.py`: Generación de embeddings y búsqueda semántica sobre `model_embeddings`.
    - `llm_gateway.py`: Abstracción multi-provider de LLM con tracking de tokens y costes.
    - `rag_service.py`: Construcción de contexto (metadatos internos) para el asistente.
    - `nlp_profiler.py`: Detección de columnas de texto, idioma y sugerencia de pipeline.
- **/worker/**: Aplicación Celery asíncrona para tareas pesadas (Train, Predict, Train_NLP, Predict_NLP, Drift Check).

## 3. Entidades Core

1. **Tenant**: Unidad lógica para suscripciones u organizaciones. Aloja las cuotas.
2. **User**: Entidad autenticada (AuthN). Pertenece a un Tenant. Soporta distintos roles (`admin`, `editor`, `viewer`) y eliminación lógica (soft delete).
3. **Dataset**: Binario ZIP junto con un `config.json` o subida declarativa. Incluye `column_types_analysis` (salida del NLP Profiler).
4. **MLModel**: Referencia en BD local del modelo. Su binario real está en MLflow y su tracker UUID (`run_id`) está asignado de MLflow.
5. **Prediction**: Registro de ejecución e inferencia con resultados, tiempos y métricas de incertidumbre (ej. varianzas epistémicas).
6. **ModelEmbedding** (Fase 1): Vector 384-d de un modelo o dataset para búsqueda semántica. Columna `embedding vector(384)` (pgvector) con índice IVFFlat.
7. **AssistantConversation** (Fase 1): Conversación del chatbot con mensajes JSON.

## 4. Workflows Críticos

### 4.1. Carga de un Modelo (Safe TorchScript / Sklearn)
1. Usuario emite un request (POST) subiendo un `.pt` (PyTorch) o un job de MLflow existente (Entrenamiento Sklearn/PyTorch).
2. FastAPI intercepta, verifica RBAC (`editor`), comprueba Cuotas (`check_model_quota`).
3. El `MLflowService` interactúa con el demonio local de MLflow: registra el modelo creando un nuevo `Run`. Si es un upload, inyecta su binario allí.
4. FastAPI guarda en la base de datos local un registro asociando el UUID MLflow de este modelo con el Tenant.

### 4.2. Flujo de Entrenamiento (Async)
1. Usuario emite un request `/training/train`.
2. El request es rate-limitado (ej. 10 peticiones / min) por `slowapi`.
3. Validaciones de Cuota de Entrenamiento y Dataset asignado.
4. FastAPI crea un registro en tabla local como `RUNNING` o despacha un `task` Celery: `worker.delay(entrenamiento_args)`. Devuelve al cliente un `Task ID`.
5. El *Celery Worker* baja de Storage (MinIO) el dataset, aplica pipelines de Preprocesamiento (`ColumnTransformer` guardados), llama a Scikit-Learn de manera paralela (`n_jobs=-1`), y sube las métricas a MLflow. Al finalizar, actualiza DB.

### 4.3. Predicción e Incertidumbre
El motor `core_ml.factory` implementa Factory Pattern para inyectar la familia de inferenciadores y de estimación de incertidumbre.
1. Se despacha Tarea Async (Single Preprocesada/Batch).
2. Se selecciona método: ej. `mc_dropout`, `tta`, `noisy_inference`, `ensemble`.
3. Celery invoca la carga dinámica en memoria del binario (vía `torch.jit.load()` si es TorchScript impidiendo Pickle vulnerabilities, o cargando Pipelines/SKLearn si es ML general).
4. El Ensemble realiza N predicciones (Forward Passes simulados perturbando datos/modelo) y fusiona resultados antes de devolverlos.

## 5. Monitorización y Servicios Avanzados

### 5.1. Data Drift (Evidently)
El sistema permite monitorizar la degradación de los modelos comparando la distribución de los datos de entrenamiento (referencia) frente a los de producción (actuales).
- **Métricas**: PSI (Population Stability Index) y Test KS (Kolmogorov-Smirnov).
- **Umbrales**: Configurables por dataset o modelo, con herencia automática.
- **Reportes**: Generación de reportes JSON detallados por cada feature.

### 5.2. Inferencia en Streaming (WebSockets)
Inferencia de baja latencia para flujos de datos continuos.
- **Autenticación**: JWT vía query param en el handshake inicial.
- **Cuotas**: Límite de 10 conexiones simultáneas por Tenant.
- **Procesamiento**: Cada mensaje recibido se preprocesa y predice en tiempo real, devolviendo también incertidumbre y XAI si se solicita.

### 5.3. Explicabilidad (XAI con SHAP)
Uso de `KernelExplainer` para proporcionar interpretabilidad agnóstica al modelo.
- **Cálculo**: Al vuelo durante la predicción (opcional) o bajo demanda vía endpoint `/explain`.
- **Visualización**: Contribuciones de cada variable al valor final de la predicción.

### 5.4. NLP Engine (Fase 1)

**Entrenamiento NLP (`nlp_trainer.py`)** — flujo asíncrono:
1. El usuario lanza `POST /api/v1/nlp/train` con `dataset_id`, `text_column`, `label_column`, `algorithm` y `task_type`.
2. La tarea Celery `run_nlp_training` carga el dataset desde Storage, entrena con HuggingFace `Trainer` (MLflow autolog en `tenant_{id}_nlp_training`), exporta a ONNX y registra el `MLModel`.
3. Métricas de clasificación (accuracy/F1/precision/recall) o NER (`seqeval`, por entidad + overall).

**Inferencia NLP (`nlp_inference.py`)**:
- ONNX Runtime si el modelo tiene `onnx_path`; si no, fallback a checkpoint transformers.
- `POST /predict` síncrono; `POST /predict/batch` vía Celery (`run_nlp_batch_inference`); `WS /streaming/{model_id}` en tiempo real.
- Devuelve `{label, confidence, entities[], tokens[], latency_ms}`.

**Semantic Search (`embedding_service.py`)**:
- Modelo `sentence-transformers/all-MiniLM-L6-v2` (384 dims) cargado lazy como singleton.
- Indexado automático al finalizar entrenamiento (`index_model`) y al subir datasets (profiler hook).
- Búsqueda: operador `<=>` (coseno) en PostgreSQL con índice IVFFlat; fallback a coseno numpy en SQLite (tests).

**Asistente LLM + RAG (`llm_gateway.py` + `rag_service.py`)**:
- Gateway multi-provider con API OpenAI-compatible (vLLM default). `stream()` emite `{delta}` y `{usage}`; `complete()` para respuestas no-stream.
- `RAGService.retrieve_context()` consulta modelos del tenant, drift reciente, costes y búsqueda semántica; el LLM **solo recibe metadatos, nunca datos crudos**.
- `POST /assistant/chat` con `stream=true` responde en SSE (`text/event-stream`). Historial persistido en `assistant_conversations`.

**NLP Profiler (`nlp_profiler.py`)**:
- Post-hook en `POST /datasets/`: detecta columnas de texto libre (heurística longitud/tokens), idioma (`langdetect`), longitud media, vocabulario estimado y sugiere pipeline (`suggested_task`, `suggested_algorithm`, `suggested_label_columns`).
- Resultado guardado en `dataset.column_types_analysis` y usado para indexar el dataset.

**Métricas Prometheus** (dominio `nlp`/`llm`): `praxisml_nlp_inference_latency_ms`, `praxisml_nlp_predictions_total`, `praxisml_embeddings_total`, `praxisml_semantic_search_latency_ms`, `praxisml_llm_tokens_total`, `praxisml_llm_requests_total`, `praxisml_llm_latency_ms`. Dashboard Grafana `praxisml-nlp` (NLP Engine).

## 6. Almacenamiento (Storage Factory)

El sistema abstrae el acceso a archivos mediante un `StorageService` que soporta múltiples backends:
- **Local**: Carpeta compartida (ideal para dev).
- **MinIO**: S3-compatible auto-alojado.
- **AWS S3**: Almacenamiento en la nube.

## 7. Control de Acceso y Rate Limiting

- **Endpoints Exigentes (Train/Pred)**: Decorador `@limiter.limit` (ej. `30/minute`). Extrae la IP de la petición limitando abuso.
- **RBAC**: Las dependencias inyectadas resuelven si un usuario tiene la facultad. El primer usuario de un Tenant adquiere cargo de `ADMIN`. Existen los roles `admin`, `editor`, `viewer`. Los `admin` pueden gestionar al resto de usuarios de su Tenant (invitación, cambio de roles, y desactivación/soft delete).

```python
# Ejemplo inyección RBAC Pydantic
@router.delete("/{id}")
def delete_model(id: UUID, current_user = Depends(require_admin)):
    ...
```

- **Quotas**:
```python
@router.post("/")
def upload_dataset(..., verify_quota = Depends(check_dataset_quota)):
    ...
```

## 8. Pruebas y CI
El proyecto cuenta con suites de `pytest` (unitarias e integración).
- Uso intensivo de *mocking* (MagicMock) en Unit tests frente a bases de MLflow o MinIO.
- Base de datos en memoria o aislada transaccionalmente para Integration tests (API REST completa cubierta en `test_api_endpoints.py`).
- Flujo CI mediante GitHub Actions validando Lints (Ruff) con un límite de cobertura exigido en CI (Pytest `--cov-fail-under=30%` alcanzando ~45%).
- **Model Validation**: Script `validate_model.py` integrado en CI para verificar que los modelos cumplen con las métricas mínimas antes de ser promovidos a producción.
