# Fase 1 — NLP Engine (Clásico + Generativo)

**Timeline:** Q1 2027 (Enero — Marzo)
**Carácter:** Vertical de producto con router propio `/api/v1/nlp/*`
**Dependencias:** Fase 0 (observabilidad transversal), infraestructura existente

> 📖 Documentación de referencia: [roadmap.md](../roadmap.md#fase-1--q1-2027-nlp-engine-clásico--generativo)

---

## Índice

- [Visión General](#visión-general)
- [Componentes](#componentes)
  - [1.1 NLP Training Pipeline](#11-nlp-training-pipeline-clásico)
  - [1.2 NLP Inference](#12-nlp-inference)
  - [1.3 Semantic Search & Embeddings](#13-semantic-search--embeddings-service)
  - [1.4 Chatbot Asistente (LLM + RAG)](#14-chatbot-asistente-llm--rag)
  - [1.5 NLP Profiler para Datasets](#15-nlp-profiler-para-datasets)
- [Stack Tecnológico](#stack-tecnológico)
- [Impacto en la Arquitectura](#impacto-en-la-arquitectura)
- [API Endpoints](#api-endpoints)
- [Riesgos y Mitigaciones](#riesgos)
- [Plan de Implementación](#plan-de-implementación)

---

## Visión General

Introducir NLP como vertical completa de la plataforma, con **dos ejes diferenciados**:

- **NLP Clásico (BERT / Transformers fine-tuned):** Clasificación de texto, NER, análisis de sentimiento — modelos entrenables dentro de PraxisML.
- **NLP Generativo (LLMs vía API / local):** Chatbot asistente RAG, resúmenes, generación de config.

Todo bajo el nuevo router `/api/v1/nlp/*`.

---

## Componentes

### 1.1 — NLP Training Pipeline (Clásico)

**Archivo:** `app/api/routes/v1/nlp/training.py`
**Servicio:** `app/services/nlp_trainer.py`

Permite al usuario subir datasets textuales (CSV con columna de texto + label) y entrenar modelos transformer fine-tuneados.

#### Endpoints

```
POST /api/v1/nlp/train              → Entrenar modelo NLP
GET  /api/v1/nlp/algorithms          → Listar arquitecturas disponibles
GET  /api/v1/nlp/training/status/{id} → Estado del entrenamiento
```

#### Arquitecturas Soportadas

| API Key | Modelo HuggingFace | Tamaño | Velocidad relativa |
|---------|-------------------|--------|-------------------|
| `distilbert` | `distilbert-base-uncased` | ~260 MB | 2x (recomendado default) |
| `bert-base` | `bert-base-uncased` | ~440 MB | 1x |
| `bert-multilingual` | `bert-base-multilingual-cased` | ~680 MB | 1x |
| `roberta` | `roberta-base` | ~500 MB | 0.9x |
| `xlm-roberta` | `xlm-roberta-base` | ~1.1 GB | 0.7x (multilingüe) |
| `biobert` | `dmis-lab/biobert-v1.1` | ~440 MB | 1x (dominio médico) |

#### Tareas Soportadas

| Task Type | Model Class | Uso |
|-----------|------------|-----|
| `text_classification` | `AutoModelForSequenceClassification` | Clasificación binaria/multiclase |
| `token_classification` | `AutoModelForTokenClassification` | NER, POS tagging |
| `sentiment_analysis` | `AutoModelForSequenceClassification` | Análisis de sentimiento |

#### Integración con MLflow

- Autologging vía `mlflow.transformers.autolog()`
- Métricas: accuracy, F1, precision, recall, loss (clasificación)
- Métricas NER: precision/recall/F1 por entidad + overall
- Parámetros: `num_train_epochs`, `learning_rate`, `per_device_train_batch_size`, `weight_decay`
- Artefactos: modelo exportado a ONNX, tokenizer, config

### 1.2 — NLP Inference

**Archivo:** `app/api/routes/v1/nlp/inference.py`

```
POST /api/v1/nlp/predict              → Clasificar texto / extraer entidades
POST /api/v1/nlp/predict/batch        → Batch inference sobre CSV
WS   /api/v1/nlp/streaming/{model_id} → Streaming NLP en tiempo real
```

Reutiliza `InferenceService` + `ModelCache` existentes. Los modelos HuggingFace se exportan a **ONNX** (`optimum`) para inferencia rápida en CPU.

#### Pipeline de Inferencia NLP

1. Tokenizar input (AutoTokenizer)
2. Forward pass (ONNX Runtime o PyTorch)
3. Post-procesar: softmax → label, o CRF → entidades
4. Devolver: `{label, confidence, entities[], tokens[], latency_ms}`

### 1.3 — Semantic Search & Embeddings Service

**Archivo:** `app/api/routes/v1/nlp/embeddings.py`
**Servicio:** `app/services/embedding_service.py`

```
POST /api/v1/nlp/embed          → Generar embeddings de texto libre
GET  /api/v1/nlp/search          → Búsqueda semántica sobre modelos/datasets
```

| Componente | Detalle |
|-----------|---------|
| **Modelo** | `sentence-transformers/all-MiniLM-L6-v2` (80 MB, CPU) |
| **Dimensión embedding** | 384 |
| **Vector Store** | PostgreSQL + `pgvector` (extensión nativa Postgres 15) |
| **Índice** | IVFFlat con `lists=100` para búsqueda aproximada |
| **Tabla** | `model_embeddings(id, tenant_id, model_id, embedding vector(384), metadata jsonb)` |

### 1.4 — Chatbot Asistente (LLM + RAG)

**Archivo:** `app/api/routes/v1/nlp/assistant.py`

Gateway a LLM (OpenAI / Anthropic / Ollama local) con patrón RAG:

1. Usuario pregunta: "¿Cómo está mi modelo de producción?"
2. El sistema construye contexto consultando la API interna (modelos, drift, métricas, costes)
3. El LLM genera respuesta basada en los metadatos (nunca recibe datos del dataset)
4. Respuesta en streaming al frontend

**Providers soportados:**

| Provider | Variable ENV | Uso |
|----------|-------------|-----|
| OpenAI | `LLM_API_KEY`, `LLM_MODEL_NAME=gpt-4-turbo` | Cloud, mejor calidad |
| Anthropic | `ANTHROPIC_API_KEY` | Cloud, contexto largo |
| Ollama (local) | `OLLAMA_BASE_URL=http://ollama:11434` | On-premise, privacidad |

### 1.5 — NLP Profiler para Datasets

Post-hook en `POST /api/v1/datasets/` que analiza automáticamente:

- Detección de columnas de texto libre (vs numéricas/categóricas)
- Idioma predominante (langdetect)
- Longitud media de textos
- Vocabulario estimado
- Sugerencia de pipeline NLP

---

## Stack Tecnológico

| Componente | Tecnología | Justificación |
|-----------|-----------|---------------|
| **NLP Clásico** | `transformers` (HuggingFace) + `datasets` | Fine-tuning de BERT/RoBERTa con Trainer API |
| **Tokenización** | `tokenizers` (HuggingFace) | Tokenización rápida (Rust), integrada con transformers |
| **Embeddings** | `sentence-transformers` (MiniLM-L6-v2) | Modelo ligero (80 MB), CPU, ideal para embeddings + profiling |
| **Vector Store** | PostgreSQL + `pgvector` | Reutiliza BD existente sin nuevo servicio |
| **LLM Gateway** | LangChain + OpenAI API ó Ollama (local) | Abstracción de provider. Ollama para deployments air-gapped |
| **NER/Entities** | `spaCy` (modelos `xx_ent_wiki_sm`) | Complemento a transformers para NER rápido sin GPU |
| **Export** | `optimum` (HuggingFace) | Exportar modelos a ONNX para inferencia rápida en CPU |

---

## Impacto en la Arquitectura

| Cambio | Detalle |
|--------|---------|
| **Nuevo paquete de rutas** | `app/api/routes/v1/nlp/` — training, inference, embeddings, assistant |
| **Nuevo servicio** | `app/services/nlp_trainer.py` — Fine-tuning de transformers |
| **Nuevo servicio** | `app/services/embedding_service.py` — Singleton SentenceTransformer |
| **Nueva extensión PostgreSQL** | `CREATE EXTENSION vector;` en `infra/postgres/` |
| **Nueva tabla** | `model_embeddings(id, tenant_id, model_id, embedding vector(384), metadata jsonb)` |
| **Nuevo campo ORM** | `Dataset.column_types_analysis: JSON` |
| **Celery task** | `app/worker/tasks/train_nlp.py` — Entrenamiento NLP asíncrono |
| **Docker Compose** | Servicio opcional `ollama` para LLM local |
| **Variables .env** | `LLM_PROVIDER`, `LLM_API_KEY`, `LLM_MODEL_NAME`, `OLLAMA_BASE_URL` |

---

## API Endpoints

### NLP Training

| Método | Path | Descripción |
|--------|------|-------------|
| POST | `/api/v1/nlp/train` | Entrenar modelo NLP |
| GET | `/api/v1/nlp/algorithms` | Listar arquitecturas disponibles |
| GET | `/api/v1/nlp/training/status/{id}` | Estado del entrenamiento |

### NLP Inference

| Método | Path | Descripción |
|--------|------|-------------|
| POST | `/api/v1/nlp/predict` | Clasificar texto / extraer entidades |
| POST | `/api/v1/nlp/predict/batch` | Batch inference sobre CSV |
| WS | `/api/v1/nlp/streaming/{model_id}` | Streaming NLP en tiempo real |

### Semantic Search

| Método | Path | Descripción |
|--------|------|-------------|
| POST | `/api/v1/nlp/embed` | Generar embeddings de texto libre |
| GET | `/api/v1/nlp/search` | Búsqueda semántica sobre modelos/datasets |

### Asistente

| Método | Path | Descripción |
|--------|------|-------------|
| POST | `/api/v1/nlp/assistant/chat` | Enviar mensaje al chatbot |
| GET | `/api/v1/nlp/assistant/history` | Historial de conversación |
| DELETE | `/api/v1/nlp/assistant/history` | Borrar historial |

---

## Riesgos

| Riesgo | Mitigación |
|--------|-----------|
| **Tamaño de modelos BERT** | BERT-base = ~440 MB. Usar DistilBERT (260 MB) como default. Export a ONNX reduce latencia 2-3x |
| **GPU para fine-tuning** | DistilBERT puede fine-tunear en CPU (~30 min para datasets pequeños). Para >10K rows, GPU recomendada |
| **Coste de API LLM** | Rate limiting por tenant (reutilizar `slowapi`), cache en Redis, semantic cache para queries repetidas |
| **Privacidad de datos** | Chatbot solo recibe metadatos, no datos. Ollama para deployments on-premise |
| **pgvector performance** | Suficiente para <100K registros. Migrar a Qdrant solo si 1M+ |
| **Calidad de respuestas LLM** | Evaluación periódica con `faithfulness_score`, `answer_relevancy`. Feedback del usuario (thumbs up/down) |

---

## Plan de Implementación

| Sub-fase | Contenido | Esfuerzo | Dependencias |
|----------|-----------|----------|-------------|
| **1.1a** | `NLPTrainer` + `hyperparams.py` NLP section + entrenamiento básico | 🟡 2-3 sem | Infraestructura GPU/CPU |
| **1.1b** | Celery task + endpoint training + MLflow autologging | 🟡 1-2 sem | 1.1a |
| **1.2** | NLP Inference + ONNX export + endpoints | 🟡 2 sem | 1.1a |
| **1.3** | EmbeddingService + pgvector + semantic search endpoints | 🟡 2 sem | Infraestructura pgvector |
| **1.4a** | LLM Gateway + LangChain + assistant endpoint | 🟡 2 sem | 1.2 (dependencia débil) |
| **1.4b** | RAG pipeline + streaming responses + history | 🔴 2-3 sem | 1.4a |
| **1.5** | NLP Profiler hook en datasets POST | 🟢 1 sem | 1.1a |

---

> [← Volver al Roadmap principal](../roadmap.md)
