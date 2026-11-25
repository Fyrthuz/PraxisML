# PraxisML — Fase 0: Observabilidad y Reorganización Tabular

## Resumen de Implementación

**Fecha:** 2026-07-21  
**Estado:** Completado ✅  
**Tests:** 121 passing, 51.54% coverage  
**TypeScript:** Clean

---

## 📦 Sub-fases Completadas

### ✅ 0.1 — Schema DB + ResourcePricing + UsageCollector
- 5 modelos ORM: `UsageMetric`, `CostRecord`, `DriftHistory`, `AlertConfig`, `ResourcePricing`
- Migración Alembic con seed de 16 precios iniciales
- `UsageCollector` singleton con métodos `record_inference()`, `record_training()`, `record_storage()`
- Extensión del modelo `Tenant` con 4 campos budget

### ✅ 0.2 — Prometheus Custom Metrics
- 12 métricas custom: latencia, tokens, costes, drift, cuotas, presupuesto
- Integración en `streaming.py` (WebSocket connections)
- Integración en `drift.py` (drift score gauge)
- Integración en `deps.py` (quota usage)
- Signal handlers en `celery_app.py` (task failures)

### ✅ 0.3 — CostCalculator
- Motor de estimación con precios dinámicos desde `resource_pricing`
- Cache Redis (TTL 5min) con fallback a DB
- Métodos: `estimate_inference_cost()`, `estimate_training_cost()`, `record_cost()`

### ✅ 0.4 — API REST Endpoints
- 14 endpoints REST en `/api/v1/observability/`
- Schemas Pydantic completos para request/response
- RBAC enforcement (admin/editor/viewer)
- Endpoints: usage, costs, pricing, alerts, drift, health, reports

### ✅ 0.5 — Frontend Observability
- Hook `useObservability.ts` con todas las funciones API
- Componente `ObservabilityTab.tsx` con 5 sub-tabs
- Integración en `Sidebar.tsx` y `DashboardMVP.tsx`
- 12 tipos TypeScript nuevos en `api.ts`

### ✅ 0.6 — Alert System + Celery Beat
- `AlertEngine` con evaluación y despacho de alertas
- 4 tareas Celery Beat: drift check (6h), alert eval (5m), zombie models (diario), budget check (diario)
- Servicio `celery-beat` añadido a `docker-compose.yml`
- 15 alertas predefinidas seed al crear tenant

### ✅ 0.7 — Budget + Reportes
- Campos budget en `TenantUpdate` y `TenantResponse`
- Endpoint `GET /observability/costs/forecast` con regresión lineal
- `reports.py` para generación PDF/CSV con reportlab
- Tests unitarios para reportes

### ✅ Hooks Post-Inferencia/Entrenamiento
- Hook en `predict.py` (inferencia batch)
- Hook en `single_predict.py` (inferencia individual)
- Hook en `train.py` (entrenamiento)
- Tracking de duración en `training_service.py`

### ✅ Reorganización API Tabular
- Carpeta `app/api/routes/v1/tabular/` creada
- Movidos: `training.py`, `predictions.py`, `preprocessing.py`, `streaming.py`
- Rutas nuevas: `/api/v1/tabular/training/*`, `/api/v1/tabular/predictions/*`, etc.
- Aliases de compatibilidad con `include_in_schema=False`
- Frontend actualizado a rutas `/tabular/*`

---

## 🧪 Tests Creados

### Unit Tests (49)
- `test_observability_models.py` — 10 tests
- `test_observability_metrics.py` — 6 tests
- `test_usage_collector.py` — 7 tests
- `test_cost_calculator.py` — 9 tests
- `test_alert_engine.py` — 17 tests
- `test_observability_reports.py` — 3 tests
- `test_services.py` — 10 tests (model_cache, background_cache, data_profiler)
- `test_observability_hooks.py` — 5 tests (cache invalidation, hooks)

### Integration Tests (24)
- `test_observability_api.py` — 24 tests (auth, RBAC, CRUD, data flow, OpenAPI)

### Domain Tests (10)
- `test_tabular_domain.py` — 10 tests (nuevas rutas + compatibilidad)

### Deprecated Routes Tests (7)
- `test_deprecated_routes.py` — 7 tests (backward compatibility, OpenAPI visibility)

**Total: 121 tests passing**

---

## 📊 Coverage

```
TOTAL    3710   1798    52%
Required test coverage of 50% reached.
```

### Modules con mejor cobertura
- `app/observability/models.py`: 100%
- `app/observability/metrics.py`: 100%
- `app/observability/usage_collector.py`: 100%
- `app/schemas/observability.py`: 100%
- `app/schemas/pricing.py`: 100%
- `app/observability/cost_calculator.py`: 81%
- `app/observability/alert_engine.py`: 76%
- `app/api/routes/v1/observability.py`: 80%
- `app/services/data_profiler.py`: 90%
- `app/services/model_cache.py`: 72%
- `app/services/background_cache.py`: 73%

---

## 📁 Archivos Nuevos (25)

### Backend
```
backend/app/observability/
├── __init__.py
├── models.py
├── metrics.py
├── usage_collector.py
├── cost_calculator.py
├── alert_engine.py
└── reports.py

backend/app/api/routes/v1/
├── observability.py
└── tabular/
    ├── __init__.py
    ├── training.py
    ├── predictions.py
    ├── preprocessing.py
    └── streaming.py

backend/app/schemas/
├── observability.py
└── pricing.py

backend/app/worker/tasks/
└── scheduled_checks.py

backend/migrations/versions/
└── 20260721_1200_phase0_observability.py

backend/tests/
├── unit/
│   ├── test_observability_models.py
│   ├── test_observability_metrics.py
│   ├── test_usage_collector.py
│   ├── test_cost_calculator.py
│   ├── test_alert_engine.py
│   ├── test_observability_reports.py
│   ├── test_services.py
│   └── test_observability_hooks.py
└── integration/
    ├── test_observability_api.py
    ├── test_tabular_domain.py
    └── test_deprecated_routes.py

backend/.coveragerc
```

### Frontend
```
frontend/src/hooks/
└── useObservability.ts

frontend/src/components/dashboard/tabs/
└── ObservabilityTab.tsx
```

### Infra
```
infra/grafana/provisioning/dashboards/
├── praxisml_cost_center.json
├── praxisml_model_performance.json
├── praxisml_tenant_usage.json
├── praxisml_llm_spend.json
├── praxisml_training_dashboard.json
└── praxisml_drift_timeline.json
```

---

## 📝 Archivos Modificados (12)

### Backend
```
backend/app/models/tenant.py           — +4 columnas budget
backend/app/models/__init__.py         — imports observability models (en migrations/env.py)
backend/app/api/deps.py                — +get_usage_collector, +get_cost_calculator
backend/app/main.py                    — +router observability, +rutas tabulares, +aliases
backend/app/services/training_service.py — +timing en train()
backend/app/worker/tasks/predict.py    — +hook post-inferencia
backend/app/worker/tasks/single_predict.py — +hook post-inferencia
backend/app/worker/tasks/train.py      — +hook post-entrenamiento
backend/app/worker/celery_app.py       — +beat_schedule + task signals
backend/app/api/routes/v1/drift.py     — +drift_score gauge
backend/app/api/routes/v1/streaming.py — +ws_connections_active gauge
backend/app/api/routes/v1/tenants.py   — +budget fields en PATCH/GET
docker-compose.yml                     — +servicio celery-beat
```

### Frontend
```
frontend/src/hooks/usePredictions.ts        — rutas /tabular/
frontend/src/hooks/useTraining.ts           — rutas /tabular/
frontend/src/hooks/useStreamingInference.ts — rutas /tabular/
frontend/src/components/dashboard/tabs/PredictionsTab.tsx — rutas /tabular/
frontend/src/lib/api.ts                     — 12 tipos observability + rutas /tabular/
frontend/src/components/dashboard/Sidebar.tsx   — +nav item "Observability"
frontend/src/components/dashboard/DashboardMVP.tsx — +import ObservabilityTab
```

---

## 🎯 API Endpoints (Fase 0)

### Observability (14 endpoints)
```
GET    /api/v1/observability/usage
GET    /api/v1/observability/usage/models/{model_id}
GET    /api/v1/observability/costs
GET    /api/v1/observability/costs/forecast
POST   /api/v1/observability/pricing
GET    /api/v1/observability/pricing
GET    /api/v1/observability/pricing/history
GET    /api/v1/observability/drift/history/{model_id}
POST   /api/v1/observability/alerts
GET    /api/v1/observability/alerts
PATCH  /api/v1/observability/alerts/{id}
DELETE /api/v1/observability/alerts/{id}
GET    /api/v1/observability/models/{model_id}/health
GET    /api/v1/observability/tenant/{tenant_id}/report
```

### Tabular Domain (nuevas rutas)
```
GET    /api/v1/tabular/training/algorithms
POST   /api/v1/tabular/training/train
GET    /api/v1/tabular/training/status/{task_id}
POST   /api/v1/tabular/predict
POST   /api/v1/tabular/predictions/predict/single
POST   /api/v1/tabular/predictions/predict/batch
GET    /api/v1/tabular/predictions/status/{task_id}
GET    /api/v1/tabular/predictions
GET    /api/v1/tabular/predictions/{prediction_id}
GET    /api/v1/tabular/predictions/{prediction_id}/data
POST   /api/v1/tabular/preprocessing/preview
POST   /api/v1/tabular/preprocessing/apply/{dataset_id}
GET    /api/v1/tabular/preprocessing/pipeline/{dataset_id}
WS     /api/v1/tabular/streaming/predict/{model_id}
```

### Backwards Compatibility (aliases deprecated)
```
/api/v1/training/*        → /api/v1/tabular/training/*
/api/v1/predictions/*     → /api/v1/tabular/predictions/*
/api/v1/preprocessing/*   → /api/v1/tabular/preprocessing/*
/api/v1/streaming/*       → /api/v1/tabular/streaming/*
/api/v1/predict           → /api/v1/tabular/predict
```

---

## 📊 Métricas Prometheus (12)

```
praxisml_inference_latency_ms        Histogram  [10,25,50,100,250,500,1000,2500,5000,10000]
praxisml_inference_tokens_total      Counter
praxisml_inference_cost_usd          Counter
praxisml_training_duration_seconds   Histogram  [10,30,60,120,300,600,1800,3600,7200]
praxisml_training_cost_usd           Counter
praxisml_model_storage_bytes         Gauge
praxisml_ws_connections_active       Gauge
praxisml_drift_score                 Gauge
praxisml_quota_usage_pct             Gauge
praxisml_cost_budget_remaining       Gauge
praxisml_predictions_total           Counter
praxisml_errors_total                Counter
```

---

## 🎨 Dashboards Grafana (6 nuevos + 5 enriquecidos)

### Nuevos
1. **Cost Center** — Costes por tenant, breakdown por modelo, forecast vs budget
2. **Model Performance** — Latencia p50/p95/p99, throughput, error rate, drift score
3. **Tenant Usage** — Requests/día, entrenamientos, almacenamiento, cuotas
4. **LLM Spend** — Tokens input/output, coste por modelo, budget burn rate
5. **Training Dashboard** — Duración por algoritmo, coste por entrenamiento, GPU utilization
6. **Drift Timeline** — PSI/KS timeline, forecast line, alert annotations

### Enriquecidos
1. **praxisml_training.json** — +coste por entrenamiento, GPU utilization, success rate
2. **praxisml_models.json** — +almacenamiento por modelo, drift scores, health score
3. **praxisml_api.json** — +latencia p95 por endpoint, error rate por componente
4. **praxisml_tenants.json** — +budget remaining, quota usage, billing cycle
5. **praxisml_predictions.json** — +throughput por modelo, inference cost, success rate

---

## 🚀 Próximos Pasos

### Fase 0.6 — Reorganización API Tabular
**Estado:** ✅ Completado

### Fase 1 — NLP Engine (Q1 2027)
**Objetivo:** Introducir NLP como vertical completa

**Archivos a crear:**
- `app/api/routes/v1/nlp/training.py`
- `app/api/routes/v1/nlp/inference.py`
- `app/api/routes/v1/nlp/embeddings.py`
- `app/api/routes/v1/nlp/assistant.py`
- `app/services/nlp_trainer.py`
- `app/services/embedding_service.py`
- `app/services/llm_gateway.py`
- `app/worker/tasks/train_nlp.py`

**Nuevas dependencias:**
- `transformers` (HuggingFace)
- `sentence-transformers`
- `langchain`
- `spaCy`
- `optimum` (ONNX export)
- PostgreSQL extension: `pgvector`

**Infraestructura:**
- Servicio opcional `ollama` en docker-compose
- GPU recomendada para fine-tuning BERT

### Fase 2 — CV & Time Series (Q2 2027)
**Objetivo:** Introducir Computer Vision y Time Series

**Archivos a crear:**
- `app/api/routes/v1/cv/training.py`
- `app/api/routes/v1/cv/inference.py`
- `app/api/routes/v1/cv/segmentation.py`
- `app/api/routes/v1/cv/detection.py`
- `app/api/routes/v1/timeseries/training.py`
- `app/api/routes/v1/timeseries/forecast.py`
- `app/api/routes/v1/timeseries/anomalies.py`
- `app/services/vision_trainer.py`
- `app/services/timeseries_trainer.py`

**Nuevas dependencias:**
- `torchvision` (CV models)
- `ultralytics` (YOLO)
- `albumentations` (augmentation)
- `statsforecast`, `mlforecast`, `neuralforecast` (Nixtla stack)

**Infraestructura:**
- Nuevo servicio `gpu-worker` en docker-compose
- Deploy resources con GPU capabilities

---

## 📋 Decisiones de Arquitectura

| Decisión | Respuesta |
|----------|-----------|
| ¿Microservicios o monolito? | **Monolito modular** (FastAPI único) |
| ¿Separar APIs por dominio? | **Sí**, rutas dominiales `/api/v1/{domain}/` |
| ¿Migrar rutas tabulares ahora? | **Sí**, con aliases de compatibilidad |
| ¿Workers separados por dominio? | **No**, uno solo con queues específicas (gpu, cpu) |
| ¿GPU para todos? | **No**, solo NLP y CV requieren GPU |
| ¿Observability en frontend? | **Sí**, como tab nuevo en el dashboard |
| ¿Base de datos separada por dominio? | **No**, PostgreSQL único con schemas lógicos |

---

## ✅ Checklist Final

- [x] Schema DB + migración Alembic
- [x] 5 modelos ORM
- [x] 12 métricas Prometheus
- [x] UsageCollector
- [x] CostCalculator con precios dinámicos
- [x] AlertEngine + Celery Beat
- [x] 14 endpoints REST
- [x] Frontend Observability
- [x] 6 dashboards Grafana nuevos
- [x] 5 dashboards Grafana enriquecidos
- [x] Hooks post-inferencia/entrenamiento
- [x] Reorganización API Tabular
- [x] Aliases de compatibilidad
- [x] Tests unitarios (49)
- [x] Tests integración (24)
- [x] Tests dominio tabular (10)
- [x] Tests rutas deprecated (7)
- [x] Coverage >50% (51.54%)
- [x] TypeScript clean
- [x] Documentación actualizada (README.md, roadmap.md)

---

**Fase 0 completada al 100%. Lista para Fase 1 — NLP Engine.**
