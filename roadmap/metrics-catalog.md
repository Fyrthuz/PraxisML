# Catálogo Completo de Métricas — PraxisML

> ~200 métricas organizadas en 13 categorías + 1 sistema de detección temprana.
> Cada modelo tipo (tabular, NLP, CV, time series, LLM) tiene su subset relevante.

---

## Índice

1. [Rendimiento de Inferencia](#1-rendimiento-de-inferencia)
2. [Métricas de Entrenamiento](#2-métricas-de-entrenamiento)
3. [Métricas de Coste](#3-métricas-de-coste-financieras)
4. [Calidad del Modelo](#4-métricas-de-calidad-del-modelo)
5. [Seguridad y Acceso](#5-métricas-de-seguridad-y-acceso)
6. [Utilización y Adopción](#6-métricas-de-utilización-y-adopción)
7. [Salud del Sistema](#7-métricas-de-salud-del-sistema)
8. [Pipeline ML (MLOps)](#8-métricas-de-pipeline-ml-mlops)
9. [Sostenibilidad](#9-métricas-ambientales-y-de-sostenibilidad)
10. [Experiencia de Usuario (RUM)](#10-métricas-de-experiencia-de-usuario-rum)
11. [Métricas Poco Convencionales](#11-métricas-poco-convencionales-pero-poderosas)
12. [LLM Específicas](#12-métricas-de-llm-específicas)
13. [Métricas Derivadas y Compuestas](#13-métricas-derivadas-y-compuestas)
14. [Sistema de Detección Temprana](#sistema-de-detección-temprana-early-warning-system)

---

## 1. Rendimiento de Inferencia

| Métrica | Tipo | Labels | Fuente |
|---------|------|--------|--------|
| `latency_p50_ms` | Histogram | tenant, domain, model_id, algorithm, task_type | InferenceService |
| `latency_p95_ms` | Histogram | ídem | InferenceService |
| `latency_p99_ms` | Histogram | ídem | InferenceService |
| `latency_max_ms` | Gauge | ídem | InferenceService |
| `latency_min_ms` | Gauge | ídem | InferenceService |
| `latency_stddev_ms` | Gauge | ídem | InferenceService |
| `latency_warm_vs_cold_ms` | Histogram | ídem + cache_hit | InferenceService + ModelCache |
| `latency_by_batch_size_ms` | Histogram | ídem + batch_size | InferenceService |
| `latency_by_input_size_ms` | Histogram | ídem + input_bytes | InferenceService + tabular_parser |
| `time_to_first_token_ms` | Histogram | model_id, provider | LLM Gateway |
| `time_between_tokens_ms` | Histogram | model_id, provider | LLM Gateway |
| `throughput_reqs_per_second` | Gauge | tenant, domain, endpoint | FastAPI auto-instrument |
| `throughput_tokens_per_second` | Gauge | model_id | LLM Gateway |
| `concurrent_requests` | Gauge | tenant | FastAPI auto-instrument |
| `queue_depth` | Gauge | tenant, queue_name | Celery |
| `queue_wait_time_ms` | Histogram | tenant, task_type | Celery |
| `timeout_rate` | Counter | tenant, endpoint | API |
| `retry_rate` | Counter | tenant, endpoint, reason | Celery Worker |

## 2. Métricas de Entrenamiento

| Métrica | Tipo | Labels | Fuente |
|---------|------|--------|--------|
| `training_duration_s` | Histogram | tenant, domain, algorithm, task_type | training_service |
| `training_gpu_hours` | Counter | tenant, gpu_type | training_service |
| `training_cpu_hours` | Counter | tenant | training_service |
| `training_epochs` | Gauge | ídem | training_service |
| `training_early_stopped` | Counter | ídem, reason | training_service |
| `training_total_flops` | Counter | ídem | Estimación |
| `training_memory_peak_gb` | Gauge | ídem | nvidia-smi / psutil |
| `training_memory_leak_rate` | Gauge | ídem | torch.cuda.memory_stats |
| `training_data_throughput` | Gauge | ídem (rows/sec o images/sec) | DataLoader |
| `training_gpu_utilization_pct` | Gauge | ídem | nvidia-smi |
| `training_gpu_memory_pct` | Gauge | ídem | nvidia-smi |
| `training_loss_convergence_slope` | Gauge | ídem | loss curve |
| `training_loss_final` | Gauge | ídem | training_service |
| `training_validation_gap` | Gauge | ídem | Calculado: train_loss - val_loss |
| `training_overfitting_score` | Gauge | ídem | (train_metric - val_metric) / train_metric |
| `training_checkpoint_size_mb` | Gauge | ídem | training_service |
| `training_export_time_s` | Histogram | ídem, export_format | training_service |
| `training_aborted_rate` | Counter | ídem, reason | Celery Worker |
| `training_trials_to_best` | Gauge | tenant, algorithm | recommendation_engine |

## 3. Métricas de Coste (Financieras)

| Métrica | Tipo | Labels | Fuente |
|---------|------|--------|--------|
| `cost_compute_usd_total` | Counter | tenant, domain, compute_type | CostCalculator |
| `cost_inference_usd` | Counter | tenant, model_id | CostCalculator |
| `cost_training_usd` | Counter | tenant, model_id | CostCalculator |
| `cost_storage_usd_monthly` | Gauge | tenant, storage_type | CostCalculator |
| `cost_api_tokens_usd` | Counter | tenant, model_id, provider | CostCalculator |
| `cost_total_monthly_usd` | Gauge | tenant | CostCalculator |
| `cost_per_inference_usd` | Gauge | model_id | CostCalculator |
| `cost_per_training_usd` | Gauge | tenant, algorithm | CostCalculator |
| `cost_per_token_usd` | Gauge | model_id, direction | CostCalculator |
| `cost_budget_used_pct` | Gauge | tenant | CostCalculator |
| `cost_budget_forecast_overrun` | Gauge | tenant | CostCalculator |
| `cost_anomaly_score` | Gauge | tenant | CostCalculator (z-score) |
| `cost_by_hour_usd` | Histogram | tenant | CostCalculator |
| `cost_infrastructure_overhead_usd` | Counter | tenant | CostCalculator |
| `roi_per_model` | Gauge | model_id | (business_value - total_cost) / total_cost |
| `cost_savings_from_cache_usd` | Counter | tenant | ModelCache |
| `cost_savings_from_early_stop_usd` | Counter | tenant | training_service |

## 4. Métricas de Calidad del Modelo

| Métrica | Tipo | Labels | Fuente |
|---------|------|--------|--------|
| `accuracy` / `f1` / `precision` / `recall` | Gauge | model_id, task_type | MLflow |
| `auc_roc` / `auc_pr` | Gauge | model_id | MLflow + training_service |
| `mse` / `mae` / `rmse` / `mape` / `smape` | Gauge | model_id | MLflow + training_service |
| `dice_coefficient` / `iou` | Gauge | model_id (segmentation) | training_service |
| `log_loss` | Gauge | model_id | training_service |
| `expected_calibration_error` | Gauge | model_id | Calculado post-inferencia |
| `brier_score` | Gauge | model_id | Calculado |
| `confidence_distribution` | Histogram | model_id | InferenceService |
| `prediction_drift_score` | Gauge | model_id, metric (psi/ks/js) | DriftService |
| `feature_drift_score` | Gauge | model_id, feature_name | DriftService |
| `concept_drift_score` | Gauge | model_id | DriftService |
| `data_quality_score` | Gauge | dataset_id | DataProfiler |
| `data_completeness_pct` | Gauge | dataset_id | DataProfiler |
| `data_freshness_days` | Gauge | dataset_id | Dataset.updated_at |
| `data_skewness` | Gauge | dataset_id, feature | DataProfiler |
| `data_class_balance_ratio` | Gauge | dataset_id | DataProfiler |
| `silhouette_score` | Gauge | model_id (clustering) | training_service |
| `faithfulness_score` | Gauge | model_id (LLM RAG) | RAG evaluator |
| `answer_relevancy` | Gauge | model_id (LLM RAG) | RAG evaluator |
| `context_precision` | Gauge | model_id (LLM RAG) | RAG evaluator |
| `hallucination_rate` | Gauge | model_id (LLM) | LLM Gateway + evaluator |
| `toxicity_score` | Gauge | model_id (LLM) | Content moderation API |
| `refusal_rate` | Gauge | model_id (LLM) | LLM Gateway |
| `bias_demographic_parity` | Gauge | model_id | Fairness evaluator |
| `bias_equal_opportunity` | Gauge | model_id | Fairness evaluator |
| `robustness_accuracy_under_perturbation` | Gauge | model_id | Test-time evaluation |
| `robustness_adversarial_accuracy` | Gauge | model_id | Adversarial evaluator |

## 5. Métricas de Seguridad y Acceso

| Métrica | Tipo | Labels | Fuente |
|---------|------|--------|--------|
| `auth_login_attempts_total` | Counter | status, method | auth.py |
| `auth_login_failure_rate` | Gauge | ip_geo | auth.py |
| `auth_token_refresh_rate` | Counter | tenant | auth.py |
| `rbac_denied_requests` | Counter | tenant, role, endpoint | deps.py |
| `api_key_usage_by_endpoint` | Counter | tenant, endpoint | security.py |
| `jwt_expired_rate` | Counter | tenant | security.py |
| `data_download_volume_bytes` | Counter | tenant, resource_type | datasets.py |
| `data_upload_volume_bytes` | Counter | tenant, resource_type | datasets.py |
| `cross_tenant_access_attempts` | Counter | source_tenant, target_tenant | deps.py |
| `webhook_failure_rate` | Counter | tenant, webhook_url | drift.py / alerts |
| `sensitive_data_exposure_attempts` | Counter | tenant | data_profiler / PII detector |
| `rate_limit_hits` | Counter | tenant, endpoint, ip | slowapi |
| `rate_limit_bypass_attempts` | Counter | ip | slowapi |
| `model_poisoning_attempts` | Counter | tenant | upload_model validation |
| `unusual_access_pattern_score` | Gauge | tenant, user_id | Anomaly detection |

## 6. Métricas de Utilización y Adopción

| Métrica | Tipo | Labels | Fuente |
|---------|------|--------|--------|
| `active_users_daily` | Gauge | tenant, role | auth.py |
| `active_users_weekly` | Gauge | tenant | auth.py |
| `active_users_monthly` | Gauge | tenant | auth.py |
| `models_in_production` | Gauge | tenant, domain | MLModel.stage |
| `models_trained_total` | Counter | tenant, domain, algorithm | training_service |
| `datasets_uploaded_total` | Counter | tenant, file_type | datasets.py |
| `predictions_total` | Counter | tenant, domain, method | predictions.py |
| `predictions_by_hour` | Histogram | tenant | predictions.py |
| `predictions_by_day_of_week` | Gauge | tenant | predictions.py |
| `unique_models_used_by_user` | Gauge | tenant, user_id | predictions.py |
| `feature_adoption_rate` | Gauge | tenant, feature | API usage |
| `dashboard_load_time_ms` | Histogram | tenant, dashboard | Frontend RUM |
| `api_endpoint_popularity` | Counter | endpoint, method | FastAPI auto-instrument |
| `experiment_success_rate` | Gauge | tenant, domain | MLflow runs |
| `experiment_abandonment_rate` | Gauge | tenant | MLflow runs |
| `time_to_first_deployment_days` | Histogram | tenant | Calculado |
| `model_lifetime_days` | Gauge | model_id | Calculado |
| `tenant_onboarding_completion_pct` | Gauge | tenant | Calculado |
| `power_users_count` | Gauge | tenant | Usuarios >100 pred/día |
| `inactive_tenants_count` | Gauge | global | Sin actividad 30 días |

## 7. Métricas de Salud del Sistema

| Métrica | Tipo | Labels | Fuente |
|---------|------|--------|--------|
| `db_connection_pool_usage_pct` | Gauge | db_name | database.py |
| `db_query_latency_ms` | Histogram | query_type | SQLAlchemy |
| `db_transaction_conflict_rate` | Counter | db_name | PostgreSQL logs |
| `db_replication_lag_bytes` | Gauge | db_name | PostgreSQL |
| `db_table_size_bytes` | Gauge | table_name | PostgreSQL |
| `db_index_bloat_pct` | Gauge | table_name, index_name | PostgreSQL |
| `redis_memory_usage_bytes` | Gauge | redis_db | Redis INFO |
| `redis_hit_rate` | Gauge | cache_name | Redis INFO |
| `redis_keys_expired_total` | Counter | redis_db | Redis INFO |
| `celery_queue_size` | Gauge | queue_name | Celery |
| `celery_worker_utilization_pct` | Gauge | worker_name | Celery |
| `celery_task_failure_rate` | Gauge | queue_name, task_name | Celery |
| `celery_task_duration_s` | Histogram | task_name | Celery |
| `celery_retry_rate` | Gauge | task_name | Celery |
| `mlflow_tracking_latency_ms` | Histogram | operation | MLflowService |
| `mlflow_artifact_storage_gb` | Gauge | tenant | MLflowService |
| `storage_backend_latency_ms` | Histogram | operation | StorageService |
| `storage_backend_throughput_mbps` | Gauge | operation | StorageService |
| `disk_usage_pct` | Gauge | mount_point | OS |
| `memory_available_bytes` | Gauge | host | OS / Docker |
| `cpu_usage_pct` | Gauge | host, core | OS / Docker |
| `docker_container_restarts_total` | Counter | container_name | Docker |
| `docker_container_oom_total` | Counter | container_name | Docker |
| `network_inbound_bytes` | Counter | host, interface | OS |
| `network_outbound_bytes` | Counter | host, interface | OS |
| `tls_certificate_expiry_days` | Gauge | hostname | Cert check |
| `dvc_sync_latency_ms` | Histogram | tenant | DVCService |
| `dvc_push_failure_rate` | Gauge | tenant, remote | DVCService |

## 8. Métricas de Pipeline ML (MLOps)

| Métrica | Tipo | Labels | Fuente |
|---------|------|--------|--------|
| `pipeline_step_duration_s` | Histogram | pipeline_id, step_name | preprocessing.py |
| `pipeline_step_memory_mb` | Gauge | pipeline_id, step_name | preprocessing.py |
| `pipeline_artifact_size_mb` | Gauge | pipeline_id | preprocessing.py |
| `pipeline_cache_hit_rate` | Gauge | tenant, pipeline_id | preprocessing.py |
| `feature_engineering_time_s` | Histogram | tenant, n_features | training_utils.py |
| `feature_importance_stability` | Gauge | model_id, feature_name | explainability.py |
| `shap_computation_time_s` | Histogram | model_id, n_features | explainability.py |
| `shap_value_distribution` | Histogram | model_id | explainability.py |
| `model_export_size_mb` | Gauge | model_id, format | training_service |
| `model_load_time_ms` | Histogram | model_id, source | InferenceService + ModelCache |
| `model_swap_rate` | Gauge | tenant | ModelCache |
| `model_version_promotions_total` | Counter | tenant, stage | models.py |
| `model_version_rollbacks_total` | Counter | tenant | models.py |
| `ci_pipeline_duration_s` | Histogram | workflow_name | GitHub Actions |
| `ci_test_failure_rate` | Gauge | test_type | GitHub Actions |
| `ci_build_size_mb` | Gauge | service | Docker build |
| `ci_model_validation_checks_failed` | Counter | check_name | model_ci.yml |
| `deploy_frequency_per_week` | Gauge | service | Git commits |
| `change_failure_rate` | Gauge | service | Rollbacks |
| `mttr_hours` | Gauge | service | Incident tracking |

## 9. Métricas Ambientales y de Sostenibilidad

| Métrica | Tipo | Labels | Fuente |
|---------|------|--------|--------|
| `energy_consumption_kwh` | Counter | tenant, compute_type | Estimación: GPU_hours × TDP |
| `carbon_footprint_kg_co2` | Counter | tenant, region | kWh × grid_intensity |
| `carbon_per_inference_g_co2` | Gauge | model_id | Estimación |
| `carbon_per_training_kg_co2` | Gauge | model_id | Estimación |
| `gpu_idle_power_watts` | Gauge | gpu_type | nvidia-smi |
| `efficiency_flops_per_watt` | Gauge | gpu_type | Calculado |
| `model_size_efficiency` | Gauge | model_id | metric / model_size |
| `inference_energy_per_sample_joules` | Gauge | model_id | Estimación |
| `compute_waste_ratio` | Gauge | tenant | aborted / total entrenamientos |
| `storage_waste_bytes` | Gauge | tenant | Modelos no usados + datasets huérfanos |
| `ephemeral_storage_usage_bytes` | Gauge | tenant | Temp directories |

## 10. Métricas de Experiencia de Usuario (RUM)

| Métrica | Tipo | Labels | Fuente |
|---------|------|--------|--------|
| `frontend_ttfb_ms` | Histogram | page, tenant | Next.js RUM |
| `frontend_lcp_ms` | Histogram | page | Next.js RUM |
| `frontend_inp_ms` | Histogram | page, component | Next.js RUM |
| `frontend_cls` | Gauge | page | Next.js RUM |
| `frontend_websocket_reconnection_rate` | Gauge | tenant | streaming.py frontend |
| `frontend_drag_drop_success_rate` | Gauge | tenant | datasets.tsx |
| `frontend_error_boundary_catches` | Counter | component, error_code | React Error Boundary |
| `frontend_session_duration_min` | Histogram | tenant, user_id | Frontend analytics |
| `frontend_feature_discovery_rate` | Gauge | feature, tenant | Frontend analytics |
| `frontend_search_query_length` | Histogram | tenant | Frontend search |
| `frontend_form_abandonment_rate` | Gauge | form_name | Frontend analytics |
| `api_response_time_perception` | Histogram | endpoint, response_size | FastAPI + Frontend metrics |
| `api_error_rate_by_user` | Gauge | tenant, user_id | FastAPI logs |
| `api_polling_efficiency` | Gauge | tenant | training_status endpoint |

## 11. Métricas Poco Convencionales pero Poderosas

| Métrica | Tipo | Labels | Fuente |
|---------|------|--------|--------|
| `model_uncertainty_calibration_curve` | Serie | model_id | uncertainty/ |
| `uncertainty_vs_error_correlation` | Gauge | model_id | Calculado |
| `prediction_magnitude_distribution` | Histogram | model_id | InferenceService |
| `prediction_entropy_timeline` | Serie | model_id | uncertainty/ |
| `data_outlier_fraction_per_batch` | Gauge | dataset_id | DataProfiler |
| `dataset_embedding_coverage` | Gauge | dataset_id | NLP profiling |
| `feature_importance_volatility` | Gauge | model_id | explainability.py |
| `model_decision_boundary_margin` | Gauge | model_id | Calculado |
| `data_memorization_score` | Gauge | model_id | Membership inference |
| `model_uncertainty_correlation_with_drift` | Gauge | model_id | Calculado |
| `data_duplication_rate` | Gauge | dataset_id | DataProfiler |
| `data_concept_evolution_speed` | Gauge | dataset_id | Embedding drift |
| `tenant_maturity_score` | Gauge | tenant | Composite score 0-100 |
| `model_complexity_ratio` | Gauge | model_id | n_params / n_samples |
| `api_contract_compliance_rate` | Gauge | endpoint | Schema validation |

## 12. Métricas de LLM Específicas

| Métrica | Tipo | Labels | Fuente |
|---------|------|--------|--------|
| `llm_tokens_input_total` | Counter | tenant, model_id, provider | LLM Gateway |
| `llm_tokens_output_total` | Counter | tenant, model_id, provider | LLM Gateway |
| `llm_tokens_input_prompt` | Histogram | model_id | LLM Gateway |
| `llm_tokens_output_completion` | Histogram | model_id | LLM Gateway |
| `llm_tokens_cached` | Counter | model_id | Semantic cache (Redis) |
| `llm_cost_per_request_usd` | Histogram | model_id, provider | CostCalculator |
| `llm_context_window_utilization_pct` | Gauge | model_id | LLM Gateway |
| `llm_time_to_first_token_ms` | Histogram | model_id, provider | LLM Gateway |
| `llm_tokens_per_second_output` | Gauge | model_id, provider | LLM Gateway |
| `llm_finish_reason_distribution` | Counter | model_id, reason | LLM Gateway |
| `llm_length_violation_rate` | Gauge | model_id | LLM Gateway |
| `llm_content_filter_hits` | Counter | model_id, filter_type | Content moderation |
| `llm_hallucination_rate` | Gauge | model_id, domain | RAG evaluator |
| `llm_faithfulness_score` | Gauge | model_id | RAG evaluator |
| `llm_answer_relevancy` | Gauge | model_id | RAG evaluator |
| `llm_context_relevancy` | Gauge | model_id | RAG evaluator |
| `llm_rag_recall` | Gauge | model_id | RAG evaluator |
| `llm_rag_precision` | Gauge | model_id | RAG evaluator |
| `llm_refusal_rate` | Gauge | model_id | LLM Gateway |
| `llm_false_refusal_rate` | Gauge | model_id | LLM Gateway |
| `llm_tool_call_success_rate` | Gauge | model_id (agent) | Copilot / LangChain |
| `llm_agent_tool_choice_distribution` | Counter | model_id, tool_name | Copilot |
| `llm_agent_steps_per_task` | Histogram | model_id | Copilot |
| `llm_feedback_user_rating` | Gauge | model_id | Frontend |
| `llm_session_duration_min` | Histogram | tenant | Chat history |
| `llm_session_depth_messages` | Histogram | tenant | Chat history |
| `llm_embedding_query_latency_ms` | Histogram | tenant | EmbeddingService |
| `llm_embedding_dimension_coverage` | Gauge | tenant | EmbeddingService |
| `llm_rate_limit_hits_by_key` | Counter | tenant, provider | LLM Gateway |
| `llm_provider_availability` | Gauge | provider | LLM Gateway health |

## 13. Métricas Derivadas y Compuestas

| Métrica | Fórmula | Qué detecta |
|---------|---------|-------------|
| **Model Health Score** | w₁·acc + w₂·(1-drift) + w₃·(1-lat_p99/baseline) + w₄·(100-cost_pct) | Score 0-100 de salud |
| **Tenant Maturity Score** | w₁·models_prod + w₂·drift_on + w₃·xai_used + w₄·pred_vol + w₅·days_active | Madurez del tenant |
| **Cost Efficiency Index** | predictions_total / total_cost_usd | Predicciones por dólar |
| **Model Performance Decay** | accuracy(t) - accuracy(t-7d) | Degradación semanal |
| **Anomaly Severity Score** | z-score(lat) + z-score(err) + z-score(tokens) | Severidad compuesta |
| **Data Freshness Risk** | 1 / (1 + days_since_update) | Riesgo de datos viejos |
| **Overfitting Index** | (train_metric - val_metric) / train_metric | Sobreajuste (0-1) |
| **Drift Leading Indicator** | uncertainty_mean(t) - uncertainty_mean(t-7d) | ¿Incertidumbre predice drift? |
| **Pipeline Bottleneck Score** | max(duration_per_step) / total_duration | Cuello de botella (0-1) |
| **API Stability Score** | 1 - (errors / total) en ventana 1h | Estabilidad actual |
| **Capacity Planning Index** | current_load / max_capacity × trend | ¿Cuándo saturaremos? |
| **ROI per Model** | (biz_value - total_cost) / total_cost | Rentabilidad |
| **Carbon Efficiency** | predictions / kg_CO2 | Predicciones por CO₂ |
| **User Engagement Score** | DAU / MAU | Stickiness |
| **Adoption Velocity** | Δ(active_models) / Δt | Velocidad de adopción |

---

## Sistema de Detección Temprana (Early Warning System)

| Escenario | Señales Tempranas | Ventana |
|-----------|-------------------|---------|
| **Drift inminente** | ↑ uncertainty + ↑ entropy + ↑ feature_volatility | 3-7 días |
| **Degradación latencia** | ↑ p99 + ↑ stddev + ↑ queue_depth | 1-2 horas |
| **OOM inminente** | ↑ memory_leak + gpu_mem > 85% sostenido | ~30 min |
| **Coste fuera de control** | ↑ tokens/output_ratio + cost_daily_avg 3 días seguidos | 2-3 días |
| **Abandono del tenant** | ↓ predictions + ↓ logins + 0 trainings en 14d | 14-30 días |
| **Underfitting** | ↓ loss_convergence_slope bajo en epochs tempranas | Durante training |
| **Calidad de datos baja** | ↑ null_rate + ↑ outlier_fraction + ↑ skewness | En subida del dataset |
| **Infraestructura saturada** | ↑ celery_queue + ↑ db_pool + ↑ redis_memory | 1-2 horas |
| **GPU infrautilizada** | gpu_util < 30% + gpu_mem < 40% | Todo el training |
| **Cache ineficiente** | cache_hit < 10% + model_swap_rate alta | Semanal |

---

> [← Volver al Roadmap principal](../roadmap.md)
