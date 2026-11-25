# PraxisML — Demo & Walkthrough Completo (Tabular + NLP + Serving + Observabilidad Total)

> **Vídeo completo (`1m 50s`, 720p H.264 Main@3.1 + AAC, ~1.7 MB):**  
> ▶️ [**Ver / Descargar `praxisml_demo.mp4`**](./praxisml_demo.mp4) *(compatible con GitHub, navegadores, VLC y WhatsApp)*

[![PraxisML Architecture & Demo Preview](./architecture_overview.png)](./praxisml_demo.mp4)

---

## Capítulos del Vídeo

| Tiempo | Sección | Qué muestra |
|---|---|---|
| `0:00 – 0:05` | **Intro & Métricas** | Resumen ejecutivo (`98+` rutas API, `11` servicios monitorizados en vivo, `18` dashboards Grafana + `5` alertas Prometheus, `7` registros pluggables). |
| `0:05 – 0:12` | **Arquitectura Modular** | Separación de *Workspace/NLP*, *Control Plane*, *Compute & Observability Stack* y *Data/Vector/MLflow Layer*. |
| `0:12 – 0:18` | **Ciclo de Vida & 7 Registros** | Flujo end-to-end Tabular + NLP ONNX y mapa de extensibilidad sin lock-in a librerías ni clouds. |
| `0:18 – 0:46` | **Overview, Datasets, Preprocessing & Environments** | 6 KPIs + barra de 10 servicios en vivo, datasets tabulares y texto (`customer_reviews_nlp`), Auto-suggest + renombrado libre de variables, `RuntimeEnvironments` (`uv`/`docker`), `def transform(df)` y **Pluggable Backends**. |
| `0:46 – 1:06` | **Models Compare & NLP Studio Completo** | Comparador lado a lado, Fine-tuning HuggingFace DistilBERT + ONNX, **Inferencia NLP en tiempo real** (label, confianza, latencia, tokens), **Búsqueda Semántica Vectorial (`pgvector`)** y **RAG Platform Copilot**. |
| `1:06 – 1:24` | **Model Endpoints (Tabular Canary + NLP ONNX)** | Métricas por endpoint (**P50/P95/P99**, success/error rate, latency timeline, tráfico Primary vs Canary ajustable en vivo, logs filtrables) e invocación real sobre `/serving/v1/titanic-live` y `/serving/v1/sentiment-nlp`. |
| `1:24 – 1:45` | **Observability Total, Governance, `Cmd+K` & MLflow** | Panel de **11 servicios y colas**, uso filtrado por dominio (`serving`, `tabular`, `nlp`, `pipelines`, `llm`), API Keys (`px_...`), Audit Log, Plugins, **Command Palette (`⌘K`)** y experimentos organizados en **MLflow UI**. |
| `1:45 – 1:50` | **Roadmap v2.0** | Fases 0–2 completadas y próximos hitos (Fase 3 Lakehouse Delta/DuckDB/Monitoring, Fases 4–5). |

---

## Capturas Clave

### Flujo End-to-End & 7 Registros Pluggables
![Lifecycle & Pluggable Registries](./lifecycle_registries.png)

### Model Serving Endpoints — Observabilidad P50/P95/P99, Canary & Live Console
![Model Serving Endpoints Preview](./gui_endpoints_preview.png)
