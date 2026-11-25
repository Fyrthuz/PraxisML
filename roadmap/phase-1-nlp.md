# Fase 1 — NLP Engine (Clásico + Semantic Search + LLM RAG)

> **Estado:** ✅ **COMPLETADA** (implementada en commits `9ff24a3` y `b0c38de`)  
> **Carácter:** Vertical de producto con router propio `/api/v1/nlp/*`  
> **Dependencias:** Fase 0 (observabilidad transversal), PostgreSQL con `pgvector`, `vLLM` (opcional, perfil `nlp`)  
> 📖 **Documentos relacionados:** [roadmap.md](../roadmap.md) · [Plan de implementación ejecutado](../docs/plan-fase-1-nlp-engine.md)

---

## Resumen de Implementación Real vs. Plan Original

Al ejecutar la Fase 1 se tomaron decisiones técnicas pragmáticas respecto al borrador inicial para mejorar estabilidad, rendimiento y compatibilidad con Python 3.10 + PyTorch ≥2.9:

| Aspecto | Borrador inicial | Implementación final en el repositorio | Motivo |
|---|---|---|---|
| **LLM Provider default** | OpenAI / Ollama | **vLLM** (default) + factory `openai` / `ollama` / `anthropic` | Alto throughput OpenAI-compatible en GPU propia |
| **SDK LLM** | LangChain | **SDK `openai` directo** (+ `anthropic` nativo) | Sin sobrecarga ni lock-in de LangChain |
| **Export ONNX** | `optimum` (HuggingFace) | **`torch.onnx.export` nativo** + `onnxruntime` | `optimum` incompatible con `torch ≥2.9` |
| **NER** | `spaCy` + transformers | **HuggingFace `AutoModelForTokenClassification` + `seqeval`** | Pipeline unificado con métricas BIO por entidad (`spaCy` aplazado) |
| **Vector Store en tests** | Solo Postgres | **`VectorType` dual** (`vector(384)` en Postgres / `JSON` en SQLite) | Permite ejecutar toda la suite CI sin servidor pgvector |
| **Observabilidad NLP** | Métricas genéricas F0 | **+7 métricas Prometheus** (`praxisml_nlp_*`, `praxisml_llm_*`) + dashboard Grafana `praxisml-nlp` | Visibilidad completa de tokens, latencia y búsqueda semántica |

---

## Componentes Implementados

### 1.1 — NLP Training Pipeline (Clásico) ✅
- **Router:** `backend/app/api/routes/v1/nlp/training.py` (`GET /algorithms`, `POST /train`, `GET /training/status/{task_id}`)
- **Servicio:** `backend/app/services/nlp_trainer.py` (`NLPTrainer`) + tarea Celery `backend/app/worker/tasks/train_nlp.py`
- **Arquitecturas:** `distilbert` (default), `bert-base`, `bert-multilingual`, `roberta`, `xlm-roberta`, `biobert`
- **Tareas:** `text_classification`, `token_classification` (NER con `seqeval`), `sentiment_analysis`
- **Tracking y Export:** MLflow (`tenant_{id}_nlp_training`), export ONNX en `backend/app/core_ml/nlp/onnx_export.py`, registro en `ml_model`, tracking de uso/coste e indexado automático de embedding.

### 1.2 — NLP Inference & Streaming ✅
- **Router:** `backend/app/api/routes/v1/nlp/inference.py` (`POST /predict`, `POST /predict/batch`, `GET /predict/batch/{id}/result`, `WS /streaming/{model_id}`)
- **Servicio:** `backend/app/services/nlp_inference.py` + tarea Celery `backend/app/worker/tasks/predict_nlp.py`
- **Motor:** ONNX Runtime en CPU cuando existe `onnx_path`, con fallback a checkpoint Transformers y caché en `ModelCache`.

### 1.3 — Semantic Search & Embeddings (`pgvector`) ✅
- **Router:** `backend/app/api/routes/v1/nlp/embeddings.py` (`POST /embed`, `GET /search`)
- **Servicio:** `backend/app/services/embedding_service.py` (`sentence-transformers/all-MiniLM-L6-v2`, 384 dims)
- **Modelo ORM:** `backend/app/models/embedding.py` (`ModelEmbedding` + `VectorType`, índice IVFFlat en PostgreSQL).

### 1.4 — Asistente LLM + RAG (Solo-Metadatos) ✅
- **Router:** `backend/app/api/routes/v1/nlp/assistant.py` (`POST /assistant/chat` JSON/SSE, `GET/DELETE /assistant/history`)
- **Servicios:** `backend/app/services/llm_gateway.py` + `backend/app/services/rag_service.py`
- **Modelo ORM:** `backend/app/models/assistant.py` (`AssistantConversation`)
- **Privacidad:** El contexto RAG se construye exclusivamente con metadatos internos (modelos, drift, costes, búsqueda semántica); **nunca se envían datos crudos de datasets al LLM**.

### 1.5 — NLP Profiler para Datasets ✅
- **Servicio:** `backend/app/services/nlp_profiler.py`
- **Integración:** Post-hook en `POST /api/v1/datasets/` que detecta columnas de texto libre, idioma (`langdetect`), vocabulario estimado y sugiere pipeline en `dataset.column_types_analysis`.

### 1.6 — Frontend, Infraestructura y Tests ✅
- **Frontend:** `useNLP.ts` + `NLPTab.tsx` (sub-tabs `NLPTrainingTab`, `NLPInferenceTab`, `SemanticSearchTab`, `AssistantTab`).
- **Infra:** Imagen DB migrada a `pgvector/pgvector:pg15`, servicio opcional `vllm` (perfil `nlp`) en `docker-compose.yml`, migración Alembic `20260815_1200_phase1_nlp.py` con seed de precios vLLM.
- **Tests:** 8 suites unitarias + 5 suites de integración dedicadas a NLP.

---

## Próxima Evolución de NLP en el Nuevo Roadmap

Aunque la vertical NLP es completamente funcional hoy, su arquitectura evolucionará en las siguientes fases:
- **Fase 2 (Modularización):** Migrar `NLPTrainer` y `NLPInferenceService` al contrato unificado `DomainPlugin`, mover sus dependencias al grupo opcional `[nlp]` de `pyproject.toml`, enrutar `train_nlp` a la cola dedicada `gpu-train` y desacoplar el NLP Profiler e indexado de embeddings mediante el **Event Bus (Redis Streams)**.
- **Fase 4 (Serving & AI Gateway v2):** Extraer la inferencia NLP al **Model Serving Plane** (con Inference Tables y traffic splitting) y evolucionar `LLMGateway` hacia **AI Gateway v2** (guardrails PII/prompt injection, semantic cache, Prompt Registry y evaluación RAG).
- **Fase 5 (Platform Copilot):** Evolucionar el asistente RAG a un Copilot agéntico con tool-calling auditado en `audit_log`.

---

> [← Volver al Roadmap principal](../roadmap.md) · [Siguiente: Fase 2 — Core Modular & Compute Plane →](phase-2-modular-core-compute.md)
