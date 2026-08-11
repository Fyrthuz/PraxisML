# Plan de Implementación — Fase 1: NLP Engine (PraxisML)

> **Alcance:** Sub-fases 1.1 → 1.5 del roadmap (`roadmap/phase-1-nlp.md`), completas.
> **Provider LLM:** vLLM (API OpenAI-compatible). **Hardware:** GPU con fallback automático a CPU.
> **Base:** Codebase actual (FastAPI monolito, Celery, MLflow, PostgreSQL+MinIO+Redis, Next.js).

## 0. Decisiones de diseño

| Decisión | Resolución |
|---|---|
| LLM Provider | **vLLM** (default). API OpenAI-compatible → SDK `openai` con `base_url=VLLM_BASE_URL`. Factory también para `openai`, `anthropic`, `ollama`. **Sin LangChain.** |
| Hardware | GPU (CUDA) con fallback automático a CPU (`torch.cuda.is_available()`). `device` parametrizable. |
| NER metrics | **`seqeval`** (F1/precision/recall por entidad + overall, formato BIO). |
| pgvector + tests | `VectorType` (SA `UserDefinedType`): `vector(384)` en PostgreSQL, `JSON` en SQLite. Búsqueda `<=>` en postgres, coseno numpy en sqlite. |
| Modelos en CI | `hf-internal-testing/tiny-random-bert`; `pytest.skip` si no hay red. LLM siempre mockeado en tests. |
| Costes | `CostCalculator` + `UsageCollector` existentes, con dominio `nlp` / `llm`. Precios de tokens leídos de `resource_pricing` por nombre de modelo. |

## 1. Dependencias nuevas

```bash
cd backend && uv add transformers datasets sentence-transformers onnxruntime onnx pgvector openai langdetect seqeval
```

| Paquete | Uso |
|---|---|
| `transformers` + `datasets` + `tokenizers` | Training e inferencia clásica (BERT/RoBERTa, NER) |
| `sentence-transformers` | Embeddings `all-MiniLM-L6-v2` (384 dims) + profiler |
| `onnxruntime` + `onnx` | Inferencia ONNX rápida en CPU |
| `pgvector` | Tipo `Vector` de SQLAlchemy |
| `openai` | Cliente OpenAI-compatible para vLLM / OpenAI / Ollama |
| `langdetect` | Detección de idioma en el NLP Profiler |
| `seqeval` | Métricas NER por entidad (BIO) |

> **Nota técnica (2026-08-11):** se reemplaza `optimum` por export ONNX **nativo con `torch.onnx.export`** (dinamismo en `input_ids`/`attention_mask`), porque `optimum` 1.x (única serie con `optimum.onnxruntime`) es incompatible con `torch 2.10` (`torch.onnx.symbolic_opset14._attention_scale` eliminado) y `optimum` 2.x eliminó `optimum.onnxruntime`. Versiones fijadas: `transformers<5` (4.57.6), `onnxruntime<1.22` (1.21.1), `onnx>=1.16,<1.18` (1.17.0) — compatibles con Python 3.10.
>
> `spaCy` queda **aplazado** (NER cubierto por `AutoModelForTokenClassification`).

## 2. Configuración (`app/core/config.py`) + `.env.example`

```python
# ── NLP / LLM ──
LLM_PROVIDER: str = "vllm"                      # vllm | openai | anthropic | ollama
VLLM_BASE_URL: str = "http://vllm:8001/v1"
VLLM_MODEL_NAME: str = "Qwen/Qwen2.5-7B-Instruct"   # DEBE coincidir con resource_id del pricing
LLM_API_KEY: str = "EMPTY"
LLM_MODEL_NAME: str = "Qwen/Qwen2.5-7B-Instruct"
OLLAMA_BASE_URL: str = "http://ollama:11434"
EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
RATE_LIMIT_LLM: str = "20/minute"
RATE_LIMIT_NLP_INFERENCE: str = "30/minute"
LLM_GPU_TYPE: str = "A100"                      # coste computacional opcional
LLM_CHARGE_COMPUTE: bool = False                # cobrar GPU-hours por request LLM
LLM_MAX_CONTEXT_TOKENS: int = 8192
```

- Añadir `RATE_LIMIT_LLM` y `RATE_LIMIT_NLP_INFERENCE` al `field_validator` de formato `N/period`.
- Añadir bloque `# ── NLP / LLM ──` a `.env.example`.
- `app/worker/celery_app.py`: `include` += `"app.worker.tasks.train_nlp"`, `"app.worker.tasks.predict_nlp"`.

## 3. Esquema de datos

### 3.1 Tabla `model_embeddings`

```
id          VARCHAR PK (uuid)
tenant_id   VARCHAR NOT NULL,  INDEX (tenant_id)
model_id    VARCHAR NULL       -- fk lógica → ml_model.id
dataset_id  VARCHAR NULL       -- fk lógica → dataset.id
embedding   VectorType(384) NOT NULL
metadata    JSON NOT NULL
created_at  DATETIME NOT NULL
```

- Regla: `model_id XOR dataset_id` (una entidad por fila). Upsert por `(tenant_id, index_type, model_id|dataset_id)`.
- Índice (solo PostgreSQL): `CREATE INDEX idx_model_embeddings_ivfflat ON model_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);`

### 3.2 Esquema `metadata` (JSON)

**Campos comunes (ambos `index_type`):**

| Campo | Tipo | Req. | Descripción |
|---|---|---|---|
| `index_type` | string | sí | `"model"` \| `"dataset"` |
| `tenant_id` | uuid | sí | Aislamiento multi-tenant |
| `embedding_model` | string | sí | `sentence-transformers/all-MiniLM-L6-v2` |
| `created_at` | ISO8601 | sí | Fecha de indexado |
| `text_preview` | string ≤200 | sí | Texto indexado truncado (UI) |
| `num_tokens_indexed` | int | sí | Tokens del texto indexado |

**`index_type: "model"`:**

| Campo | Tipo | Req. | Descripción |
|---|---|---|---|
| `model_id` | uuid | sí | |
| `model_name` | string | sí | Nombre visible |
| `description` | string | op | |
| `domain` | string | sí | `"nlp"` \| `"tabular"` |
| `algorithm` | string | sí | `distilbert`, `bert-base`, `roberta`, ... |
| `task_type` | string | sí | `text_classification` \| `token_classification` \| `sentiment_analysis` |
| `framework` | string | sí | `"huggingface"` |
| `stage` | string | sí | `Staging` \| `Production` \| `Archived` |
| `mlflow_run_id` | string | sí | Para cargar el modelo |
| `mlflow_registry_name` | string | op | |
| `mlflow_version` | string | op | |
| `version` | string | op | Semantic version |
| `text_column` | string | op | Columna texto del training |
| `label_column` | string | op | Columna target del training |
| `label_map` | object | op | `{"neg":0,"pos":1}` |
| `metrics` | object | op | `{"test_accuracy":0.91,"test_f1":0.90,...}` |
| `dataset_id` | uuid | op | Dataset origen |
| `dataset_name` | string | op | |

**`index_type: "dataset"`:**

| Campo | Tipo | Req. | Descripción |
|---|---|---|---|
| `dataset_id` | uuid | sí | |
| `name` | string | sí | |
| `description` | string | op | |
| `file_type` | string | sí | `csv`, `xlsx`, `parquet` |
| `num_rows` / `num_columns` | int | sí | |
| `column_names` | string[] | sí | |
| `domain` | string | sí | `"tabular"` |
| `version` | int | op | |
| `column_types_analysis` | object | op | Salida completa del profiler 1.5 (§8.5) |

**Consumidores del esquema:** UI (tarjetas con `text_preview` + badges) · RAG assistant (1.4b: `metrics`, `stage`, `column_types_analysis`) · Recommender (Fase 3: similitud de datasets).

### 3.3 Tabla `assistant_conversations`

```
id           VARCHAR PK (uuid)
tenant_id    VARCHAR NOT NULL,  INDEX
user_id      VARCHAR NULL
messages     JSON NOT NULL      -- [{role, content, ts}]
created_at   DATETIME NOT NULL
updated_at   DATETIME NOT NULL
```

### 3.4 Columna nueva en `dataset`

`column_types_analysis JSON NULL` — poblada por el NLP Profiler en `POST /datasets/`.

## 4. Migración Alembic

Archivo: `migrations/versions/20260815_1200_phase1_nlp.py`

1. `CREATE EXTENSION IF NOT EXISTS vector` (guard por dialecto).
2. Crear `model_embeddings` (§3.1) + índice IVFFlat (solo postgres).
3. Crear `assistant_conversations` (§3.3).
4. `ALTER TABLE dataset ADD COLUMN column_types_analysis JSON`.
5. Seed de precios vLLM (§5).
6. `migrations/env.py`: importar `app.models.embedding` y `app.models.assistant`.
7. `app/schemas/dataset.py`: `column_types_analysis: Optional[dict]` en `DatasetResponse`.

## 5. Precios vLLM

**Modelo de coste (2 componentes):**

| Componente | Fuente en BD | Cálculo | Default |
|---|---|---|---|
| **Tokens** | `resource_pricing` categoría `llm_tokens`, `resource_id` = `VLLM_MODEL_NAME` | `estimate_inference_cost(domain="llm", tokens={"model": VLLM_MODEL_NAME, "input": n, "output": n})` → `(input/1000)·p_in + (output/1000)·p_out` | **0.0** (self-hosted) |
| **Compute (opcional)** | `resource_pricing` categoría `compute_gpu`, `resource_id` = `LLM_GPU_TYPE` | `latency_s/3600 · price` si `LLM_CHARGE_COMPUTE=true` | `false` |

**Seed en la migración:**
```python
("llm_tokens", "Qwen/Qwen2.5-7B-Instruct", "per_1k_input_tokens", 0.0),
("llm_tokens", "Qwen/Qwen2.5-7B-Instruct", "per_1k_output_tokens", 0.0),
```

**Gestión (sin código nuevo):** `POST /observability/pricing` (admin) para crear/actualizar precio (cierra el efectivo anterior); `GET /observability/pricing?category=llm_tokens` para consultar. El gateway lee el precio en cada llamada (cache Redis TTL 300s en `CostCalculator.get_price`).

**Registro:** en `LLMGateway._record_usage`: `record_cost(cost_type="llm", amount=tokens+compute, source=f"vllm:{VLLM_MODEL_NAME}", metadata={provider, model, tokens_input, tokens_output, latency_ms})`.

## 6. Observabilidad

### 6.1 Métricas Prometheus nuevas (`app/observability/metrics.py`)

| Métrica | Tipo | Labels | Origen |
|---|---|---|---|
| `llm_tokens_total` | Counter | tenant_id, model_id, provider, direction | LLM Gateway |
| `llm_requests_total` | Counter | tenant_id, model_id, provider, status | LLM Gateway |
| `llm_latency_ms` | Histogram | tenant_id, model_id, provider | LLM Gateway |
| `nlp_inference_latency_ms` | Histogram | tenant_id, model_id, task_type | NLP inference |
| `nlp_predictions_total` | Counter | tenant_id, model_id, task_type, status | NLP inference |
| `semantic_search_latency_ms` | Histogram | tenant_id, index_type | EmbeddingService |
| `embeddings_total` | Counter | tenant_id, operation (`embed\|index\|search`) | EmbeddingService |

Reutilizadas (ya existen): `inference_latency_ms` (label `domain=nlp/llm`), `inference_tokens_total`, `training_duration_seconds` (`domain=nlp`), `training_cost_usd`, `model_storage_bytes`, `predictions_total`, `errors_total`.

### 6.2 `UsageCollector` por evento

| Evento | Llamada |
|---|---|
| NLP training | `record_training(domain="nlp", duration_s, cpu/gpu_hours, tags={algorithm, task_type})` |
| NLP inference (predict/batch/WS) | `record_inference(domain="nlp", latency_ms, tokens_input=<n>, tags={algorithm, task_type})` |
| LLM chat | `record_inference(domain="llm", latency_ms, tokens_input, tokens_output, tags={provider, model})` |
| Embeddings | `usage_metric` ad-hoc: `embeddings_count`, `search_latency_ms` |

### 6.3 `CostRecord` por evento

| Evento | `cost_type` | Cálculo |
|---|---|---|
| NLP training | `training` | `estimate_training_cost(gpu_hours, gpu_type\|None, cpu_hours)` |
| NLP inference | `inference` | `estimate_inference_cost(domain="nlp", latency_ms, gpu_type\|None)` |
| LLM chat | `llm` | tokens + compute opcional (§5) |

### 6.4 Grafana

- Nuevo dashboard `infra/grafana/provisioning/dashboards/praxisml_nlp.json`: NLP training (duración/coste), NLP inference (latencia, tokens, predicciones por task_type), embeddings ops, LLM tokens/cost por tenant+model. El coste token adicional se sigue viendo en `praxisml_llm_spend.json` existente.

## 7. Infraestructura Docker

- `db`: imagen **`pgvector/pgvector:pg15`**.
- `infra/postgres/init-mlflow-db.sh`: añadir `CREATE EXTENSION IF NOT EXISTS vector;`.
- Servicio `vllm` (opcional, `profiles: ["nlp"]`, requiere GPU):

```yaml
vllm:
  image: vllm/vllm-openai:latest
  profiles: ["nlp"]
  ports: ["8001:8001"]
  environment: { VLLM_MODEL: "${VLLM_MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}" }
  command: ["--host", "0.0.0.0", "--port", "8001"]
  volumes: ["vllm_cache:/root/.cache/huggingface"]
  deploy:
    resources: { reservations: { devices: [ { capabilities: [gpu] } ] } }
```

- Añadir `vllm_cache:` al bloque `volumes`.

## 8. Backend por sub-fase

### 8.1 Sub-fase 1.1a — NLP Training (clásico)

**`app/core_ml/hyperparams.py`** — sección `NLP_ALGORITHMS` (mismo contrato que `get_all_algorithms()`):

| API Key | HF Model | Task types |
|---|---|---|
| `distilbert` (default) | `distilbert-base-uncased` | text_classification, token_classification, sentiment_analysis |
| `bert-base` | `bert-base-uncased` | ídem |
| `bert-multilingual` | `bert-base-multilingual-cased` | ídem |
| `roberta` | `roberta-base` | ídem |
| `xlm-roberta` | `xlm-roberta-base` | ídem |
| `biobert` | `dmis-lab/biobert-v1.1` | ídem |

Hiperparámetros configurables: `epochs` (1–20, dflt 3), `learning_rate` (1e-5–1e-3, dflt 2e-5), `batch_size` (4–64, dflt 16), `weight_decay` (0–0.1, dflt 0.01), `warmup_ratio` (0–0.5, dflt 0.1), `max_seq_length` (32–512, dflt 128).
Helpers: `get_nlp_algorithm_info()`, `get_all_nlp_algorithms()`, `get_nlp_default_hyperparams()`.

**`app/services/nlp_trainer.py`** — `NLPTrainer(tenant_id, mlflow_tracking_uri=None, device=auto)`:
- `PRETRAINED_MODELS` (6) y `TASK_TYPES` (`AutoModelForSequenceClassification` ×2, `AutoModelForTokenClassification`).
- `train(df, text_column, label_column, algorithm, task_type, hyperparams, validation_config, model_name)`:
  1. Validar columnas, filtrar filas vacías, `label_map`.
  2. `AutoTokenizer`, `max_seq_length`, padding/truncation.
  3. `torch Dataset` + split estratificado (holdout) o K-fold (`validation.strategy`).
  4. `TrainingArguments(output_dir=data/tenants/{tenant}/nlp_models/, ...)` + `Trainer` + `mlflow.transformers.autolog()`.
  5. `compute_metrics`: clasificación → `accuracy/f1/precision/recall` (weighted); NER → **seqeval** (`classification_report` por entidad + overall).
  6. Guardar modelo+tokenizer (`save_pretrained`) + `mlflow.transformers.log_model`.
  7. **Export ONNX** (`app/core_ml/nlp/onnx_export.py`): `torch.onnx.export` (opset 14, ejes dinámicos `input_ids`/`attention_mask`, dtype int64) → `.onnx` + tokenizer + `mlflow.log_artifact`.
  8. Devuelve `{mlflow_run_id, model_path, tokenizer_path, onnx_path, label_map, metrics, feature_names:[text_column], duration_s, device}`.

### 8.2 Sub-fase 1.1b — Celery + endpoints + MLflow

**`app/worker/tasks/train_nlp.py::run_nlp_training(...)`** (espejo de `run_training`): cargar dataset (storage + `read_tabular`) → `NLPTrainer.train` → crear `MLModel` con `metrics_metadata` (domain=`nlp`, onnx_path, label_map, ...) → Registry si `registry_name` → observabilidad (§6) → `EmbeddingService.index_model` (best-effort).

**`app/api/routes/v1/nlp/training.py`:**
- `GET /algorithms` → `get_all_nlp_algorithms()` (viewer).
- `POST /train` → `TrainRequest{dataset_id, text_column, label_column, algorithm, task_type, hyperparams, validation, model_name, registry_name}`. Valida algoritmo/task_type/dataset tabular/columnas. Lanza `run_nlp_training.delay(...)`. (editor + `check_training_quota` + `limiter.limit(RATE_LIMIT_TRAINING)`).
- `GET /training/status/{task_id}` → `AsyncResult` (viewer).

**`main.py`**: registrar 4 routers NLP con `prefix=f"{API_V1_STR}/nlp"`, `tags=["NLP"]`.

### 8.3 Sub-fase 1.2 — NLP Inference + ONNX

**`app/services/nlp_inference.py` — `NLPInferenceService`:**
- `load_model(ml_model, device)`: ONNX si `onnx_path`, si no `mlflow.transformers.load_model`; cache en `ModelCache`.
- `predict(model, tokenizer, text, task_type, label_map, id2label)` → tokenizar → forward → softmax → `{label, confidence}` o BIO→`entities[]`; más `tokens[]`, `latency_ms`.
- `predict_batch(df, text_column, ...)`.
- `record_usage(...)` → `UsageCollector` + `CostCalculator` (§6).

**`app/api/routes/v1/nlp/inference.py`:**
- `POST /predict` → `PredictRequest{model_id, text}` → JSON `{label, confidence, entities[], tokens[], latency_ms}` (editor + `check_prediction_quota` + `RATE_LIMIT_INFERENCE`).
- `POST /predict/batch` → multipart `(model_id, text_column, file csv)` → `Prediction` PENDING + tarea `app/worker/tasks/predict_nlp.py::run_nlp_batch_inference` → 202 `{prediction_id, task_id, status_url}`.
- `WS /streaming/{model_id}` → `?token=JWT`, patrón `tabular/streaming.py` (cuota 10/tenant Redis+in-memory, `decode_token`); recibe `{text}` → `send_json(resultado)`.

### 8.4 Sub-fase 1.3 — Embeddings + pgvector + Semantic Search

**`app/models/embedding.py`:** `VectorType(SA.UserDefinedType)` (`get_col_spec`: `vector(384)` / `JSON`; bind/result processors) + `ModelEmbedding` (§3.1).

**`app/services/embedding_service.py` — singleton `EmbeddingService`:**
- Lazy `SentenceTransformer(EMBEDDING_MODEL_NAME)` (cacheado).
- `embed_texts(texts) → np.ndarray (n,384)`.
- `index_model(db, ml_model)` / `index_dataset(db, dataset, column_types_analysis)` → **upsert** por clave de entidad.
- `search(db, tenant_id, query, top_k=20, index="models"|"datasets")` → postgres `<=>` (coseno, IVFFlat); sqlite coseno numpy. Filtra por tenant + tipo.

**`app/api/routes/v1/nlp/embeddings.py`:**
- `POST /embed` → `{texts}` → `{dim:384, model, embeddings}` (viewer).
- `GET /search?q=&index=models&limit=20` → `{results:[{model_id|dataset_id, index_type, score, metadata}]}` (viewer + `get_current_tenant`).

### 8.5 Sub-fase 1.4a — LLM Gateway (vLLM)

**`app/services/llm_gateway.py` — `LLMGateway`:**
- Factory por `LLM_PROVIDER`: `vllm`/`openai` → `AsyncOpenAI(base_url=VLLM_BASE_URL, api_key=LLM_API_KEY)`; `ollama` → `AsyncOpenAI(base_url=OLLAMA_BASE_URL)`; `anthropic` → `AsyncAnthropic`.
- `async complete(messages, system, max_tokens=1024, temperature=0.2) -> {content, usage}`.
- `async stream(messages, system, ...) -> AsyncGenerator[{delta}]`.
- `_record_usage(db, tenant_id, model_id, usage, latency_ms)` → observabilidad + precios (§5/§6).

**`app/api/routes/v1/nlp/assistant.py`:**
- `POST /assistant/chat` → `ChatRequest{message, conversation_id?}` → JSON o SSE según `stream` (editor + `limiter.limit(RATE_LIMIT_LLM)`).
- `GET /assistant/history` → historial paginado del tenant (viewer).
- `DELETE /assistant/history` → borrar (editor).

### 8.6 Sub-fase 1.4b — RAG + streaming + history

**`app/services/rag_service.py` — `RAGService`:**
- `_retrieve_context(db, tenant_id, question)`: modelos del tenant (stage + métricas), `DriftHistory` reciente, costes (`CostRecord`), `EmbeddingService.search` top-5.
- `_build_messages(user_message, context)`: `system_prompt` que declara **"solo tienes acceso a metadatos, nunca a datos crudos de los datasets"** + contexto en JSON.
- `chat(stream=True)` → `StreamingResponse` SSE (`text/event-stream`): eventos `{"delta": ...}` + cierre `{"sources", "tokens", "conversation_id"}`; `stream=False` → JSON.
- Persistir mensajes en `assistant_conversations`.

### 8.7 Sub-fase 1.5 — NLP Profiler

**`app/services/nlp_profiler.py` — `NLPProfiler.analyze_df(df, column_names=None, sample=2000)` →**
```jsonc
{
  "text_columns": ["review"],
  "numeric_columns": ["rating"],
  "categorical_columns": [],
  "predominant_language": "en",
  "avg_text_length_chars": 132.4,
  "avg_tokens_per_text": 28.1,
  "estimated_vocab_size": 1843,
  "suggested_task": "sentiment_analysis",
  "suggested_algorithm": "distilbert",
  "suggested_label_columns": ["rating"]
}
```
Detección de columnas texto (dtype object, longitud media, cardinalidad), idioma con `langdetect` sobre muestra.

**Hook en `app/api/routes/v1/datasets.py` → `POST /datasets/`**: tras crear el dataset, si `is_tabular`:
```python
try:
    analysis = NLPProfiler().analyze_df(df, column_names)
    new_dataset.column_types_analysis = analysis
    db.commit()
    EmbeddingService().index_dataset(db, new_dataset, analysis)  # best-effort
except Exception as e:
    logger.warning("NLP profiler skipped: %s", e)
```

## 9. Frontend (Next.js)

- **`src/lib/api.ts`**: tipos `NLPAlgorithm`, `NLPTrainingRequest/Status`, `NLPPrediction{label,confidence,entities,tokens,latency_ms}`, `EmbeddingResponse`, `SemanticSearchResult`, `ChatRequest/Response/StreamEvent`, `ChatHistoryItem`, `ColumnTypesAnalysis`; `API_NLP = config.getFullApiUrl("/nlp")`.
- **`src/hooks/useNLP.ts`**: `fetchNLPAlgorithms`, `startNLPTraining` + polling 3s, `predictNLP`, `batchPredictNLP`, `embedText`, `semanticSearch`, `sendChat` (SSE con `ReadableStream`), `fetchChatHistory`, `clearChatHistory`, `sendFeedback`.
- **`src/components/dashboard/tabs/NLPTab.tsx`** (patrón `ObservabilityTab`) con sub-tabs:
  - `NLPTrainingTab.tsx` — dataset + `text_column`/`label_column` (de `column_names`) + algoritmo (`/nlp/algorithms`) + hiperparámetros dinámicos + estado job.
  - `NLPInferenceTab.tsx` — modelo NLP (filtro `useModels` por `domain:"nlp"`) + textarea → label/entities/tokens/latency; batch CSV con descarga de resultados.
  - `SemanticSearchTab.tsx` — query + toggle `models|datasets` → resultados (nombre, score, `text_preview`, badges).
  - `AssistantTab.tsx` — chat burbujas con streaming, `sources` expandibles, thumbs up/down, borrar historial.
- **`Sidebar.tsx`**: `View` += `"nlp"`; item `{id:"nlp", label:"NLP", icon:<Sparkles/>}`.
- **`DashboardMVP.tsx`**: `View` += `"nlp"` + render `NLPTab`.

## 10. Tests

**Unitarios** (`backend/tests/unit/`, mocks, sin red ni DB):
| Archivo | Cubre |
|---|---|
| `test_nlp_hyperparams.py` | 6 algoritmos, task_types, defaults, `ValueError` |
| `test_nlp_trainer.py` | mock HF/`Trainer`/mlflow/onnx: clasificación, NER (seqeval), label_map, holdout/CV, export ONNX (`torch.onnx.export`) |
| `test_nlp_inference.py` | tokenize→forward→softmax→label/entities, formato output, `latency_ms`, cpu/cuda |
| `test_embedding_service.py` | dims 384, upsert index, search sqlite brute-force, aislamiento tenant |
| `test_vector_type.py` | compila `vector(384)` en postgres / `JSON` en sqlite |
| `test_llm_gateway.py` | mock `AsyncOpenAI`: complete/stream, tokens, factory por provider, pricing |
| `test_rag_service.py` | context retrieval, system prompt sin datos crudos, persistencia conversación |
| `test_nlp_profiler.py` | detección de columnas, idioma, longitud, sugerencias |

**Integración** (`backend/tests/integration/`, SQLite + override `get_db`, patrón `test_tabular_domain.py`):
| Archivo | Cubre |
|---|---|
| `test_nlp_training_api.py` | `GET /nlp/algorithms`, `POST /nlp/train` (mock tarea), `GET /nlp/training/status/{id}`, 400/404, RBAC |
| `test_nlp_inference_api.py` | `POST /nlp/predict` (mock servicio), batch 202+task, WS streaming |
| `test_nlp_embeddings_api.py` | `POST /nlp/embed`, `GET /nlp/search` con filas seed sqlite |
| `test_nlp_assistant_api.py` | chat JSON+SSE (mock gateway), history, rate limit |
| `test_nlp_profiler_hook.py` | `POST /datasets/` CSV textual → `column_types_analysis` en respuesta |

Revisar `test_main_app.py` para que el OpenAPI con los 4 routers nuevos no rompa tests existentes.

## 11. Orden de ejecución

| Paso | Contenido | Depende |
|---|---|---|
| 1 | Deps (incl. seqeval) + config + `.env.example` + celery includes | — |
| 2 | `VectorType` + `ModelEmbedding` + `AssistantConversation` + migración + seed precios vLLM + schema dataset | 1 |
| 3 | 1.1a: hyperparams NLP + `nlp_trainer.py` + `onnx_export.py` + unit tests | 1 |
| 4 | 1.1b: `train_nlp.py` + `nlp/training.py` + `main.py` + integración API | 3 |
| 5 | 1.2: `nlp_inference.py` + `predict_nlp.py` + `nlp/inference.py` + WS + tests | 3 |
| 6 | 1.3: `embedding_service.py` + `nlp/embeddings.py` + hooks indexado + tests | 2, 4 |
| 7 | 1.4a: `llm_gateway.py` + `assistant.py` + tests | 1 |
| 8 | 1.4b: `rag_service.py` + SSE + history + tests | 7, 6 |
| 9 | 1.5: `nlp_profiler.py` + hook datasets + tests | 3, 2 |
| 10 | Docker (pgvector image, vllm, init script) + métricas Prometheus + dashboard Grafana | paralelo |
| 11 | Frontend: tipos + `useNLP` + `NLPTab` + 4 sub-tabs + Sidebar + DashboardMVP | 4, 5, 6, 8 |
| 12 | pytest + ruff + tsc | todo |

## 12. Verificación final

```bash
cd backend && uv run pytest && uv run ruff check app tests
cd frontend/frontend && npx tsc --noEmit
docker compose -f docker-compose.yml --profile nlp up -d   # entorno con vLLM
```
