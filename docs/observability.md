# Observabilidad, FinOps, Grafana y CI/CD

## 1. Métricas Prometheus (`/metrics`)

La API expone las métricas HTTP de `prometheus-fastapi-instrumentator` (`http_requests_total`, `http_request_duration_seconds_bucket`) más **25 métricas custom** definidas en `backend/app/observability/metrics.py` e hidratadas al arrancar vía `backend/app/observability/bootstrap.py`:

### Core ML, Costes, Cuotas y Drift
- `praxisml_inference_latency_ms` (Histogram)
- `praxisml_inference_tokens_total` (Counter)
- `praxisml_inference_cost_usd_total` (Counter — `inference`, `serving`, `api_tokens`)
- `praxisml_predictions_total` (Counter)
- `praxisml_training_duration_seconds` (Histogram)
- `praxisml_training_cost_usd_total` (Counter)
- `praxisml_model_storage_bytes` (Gauge)
- `praxisml_ws_connections_active` (Gauge)
- `praxisml_drift_score` (Gauge — PSI/KS)
- `praxisml_quota_usage_pct` (Gauge)
- `praxisml_cost_budget_remaining` (Gauge)
- `praxisml_errors_total` (Counter)

### NLP & LLM Gateway
- `praxisml_nlp_inference_latency_ms` (Histogram)
- `praxisml_nlp_predictions_total` (Counter)
- `praxisml_embeddings_total` (Counter — `embed`, `index`, `search`)
- `praxisml_semantic_search_latency_ms` (Histogram)
- `praxisml_llm_tokens_total` (Counter)
- `praxisml_llm_requests_total` (Counter)
- `praxisml_llm_latency_ms` (Histogram)

### Model Serving, User Pipelines e Infraestructura
- `praxisml_serving_requests_total` (Counter — por `endpoint_slug`, `variant`, `status`)
- `praxisml_serving_latency_ms` (Histogram — por `endpoint_slug`, `runtime`)
- `praxisml_pipeline_runs_total` (Counter — por `engine`, `status`)
- `praxisml_pipeline_duration_seconds` (Histogram — por `engine`)
- `praxisml_service_health` (Gauge — estado de los 11 servicios: `1=ok`, `0.5=idle/rag_local`, `0=offline`)
- `praxisml_celery_queue_depth` (Gauge — colas `cpu-light`, `cpu-heavy`, `gpu-train`, `system-beat`, `celery`)

---

## 2. Scrape Targets, Alertas y 18 Dashboards Grafana

- **Targets scrapeados** (`infra/prometheus/prometheus.yml`):
  - `praxisml_api` (`http://api:8000/metrics`)
  - `minio` (`http://minio:9000/minio/v2/metrics/cluster`)
  - `grafana` (`http://grafana:3000/metrics`)
  - `prometheus` (`http://localhost:9090/metrics`)
- **Reglas de alerta** (`infra/prometheus/alerts.yml`): 5 reglas activas sobre errores de serving, latencia P95 de endpoints, disponibilidad de servicios, fallos de pipelines y errores 5xx de API.
- **18 Dashboards provisionados** (`infra/grafana/provisioning/dashboards/`):
  - **PraxisML — Global Overview**: RPS/latencia HTTP p50/p95, errores 4xx/5xx, Prometheus Targets UP, salud de los 11 servicios, objetos/buckets MinIO, Serving RPS y User Pipelines.
  - **PraxisML — Model Serving & User Pipelines**: invocaciones totales, error rate, P50/P95/P99 por endpoint/runtime, Primary vs Canary y duración de pipelines.
  - **FinOps & ML**: `Cost Center`, `LLM Spend`, `Model Performance`, `Tenant Usage`, `Training Dashboard`, `Drift Timeline`, `NLP Engine`.
  - **Por dominio de rutas API**: `api`, `auth`, `datasets`, `models`, `predictions`, `preprocessing`, `profiling`, `tenants`, `training`.

---

## 3. FinOps y Estimación Mensual de Costes

- **Precios dinámicos**: almacenados en `resource_pricing` (GPU A100/V100/T4/L4, CPU standard, storage GB/mes y tokens LLM por modelo) con caché Redis (TTL 5 min) y gestión vía `GET/POST /api/v1/observability/pricing`.
- **Registro de costes (`CostCalculator`)**: captura `training`, `inference`, `serving` y `llm`, actualiza `praxisml_inference_cost_usd_total` / `praxisml_training_cost_usd_total` y el gauge de presupuesto restante (`praxisml_cost_budget_remaining`).
- **Estimación mensual (`GET /api/v1/observability/costs/forecast`)**:
  - Con `1–6` días de histórico: proyección a fin de mes por run-rate diario (`month_to_date + avg_daily * remaining_days`).
  - Con `≥ 7` días de histórico: regresión lineal acumulada sobre los días restantes del mes + cálculo de `overrun_risk_pct` frente a `monthly_budget_usd`.
- **Reportes**: exportación mensual en PDF o CSV vía `GET /api/v1/observability/tenant/{id}/report`.

---

## 4. Pipelines CI/CD (GitHub Actions)

### CI Principal (`.github/workflows/ci.yml`)

| Job / Paso | Comando |
|------------|---------|
| **Lint** | `ruff check app/ --select=E,F,W --ignore=E501` |
| **Tests unitarios** | `pytest tests/unit/ -v` |
| **Tests de integración** | `pytest tests/integration/ -v` |
| **Cobertura** | `pytest tests/ --cov=app --cov-fail-under=50` |

### Model CI (`.github/workflows/model_ci.yml`)
Valida las métricas de un `run_id` de MLflow contra thresholds definidos y promueve automáticamente a `Production` si supera los criterios:

```bash
gh workflow run model_ci.yml -f run_id=<MLFLOW_RUN_ID> -f target_stage=Production
```
