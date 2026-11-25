# Frontend Documentation (Next.js)

El cliente principal de PraxisML, una SPA (Single Page Application) servida desde servidor, construida con los últimos estándares del ecosistema React.

## 1. Stack Tecnológico
- **Framework Web**: `Next.js 16` (App Router `src/app/`).
- **Librería UI**: `React 19`.
- **Ecosistema CSS**: `TailwindCSS v4` completo implementando `postcss`.
- **Componentes Base**: `Radix UI` proporcionando accesibilidad y modales primitivos limpios (e.g. `lucide-react` para iconos gráficos).
- **Visualización Avanzada**: `chart.js` y `react-chartjs-2` para gráficos de explicabilidad y drift.
- **Tipado**: `TypeScript 5` estricto de end-to-end con `eslint-config-next`.
- **Notificaciones/Alertas**: `react-hot-toast` para propagación de errores provenientes de tareas asíncronas de Celery o WebSockets.
- **Charts y Gráficos**: `recharts` para visualizaciones generales y paneles del tenant.

## 2. Estilo de Código y Directrices Generales

- **App Router (`src/app`)**: Navegación principal. Incluye secciones para `Datasets`, `Models`, `Predictions`, `Streaming`, `Observability` y la nueva sección de `NLP`.
- **Auth Provider**: El patrón context encapsulado, persistiendo o refrescando el token de `FastAPI` transparentemente (OAuth2).
- **Custom Hooks (`src/hooks`)**: Lógica encapsulada para funcionalidades complejas:
    - `useDrift`: Gestión de reportes de drift y actualización de umbrales.
    - `useStreamingInference`: Manejo del ciclo de vida de WebSockets para inferencia en tiempo real.
    - `usePredictions`: Polling y resultados de inferencia batch/single.
    - `useObservability`: Métricas de uso, costes, alertas y pricing (Fase 0).
    - `useNLP`: Algoritmos NLP, entrenamiento (polling), inferencia, embeddings, búsqueda semántica y chat SSE (Fase 1).
- **Drag & Drop**: Funciones para la carga local de Datasets (`ZIP`, `.json`) y modelos seguros de Pytorch (`.pt`).
- **Manejo de Errores de API (4xx / 5xx)**: Interceptor global que refleja errores en alertas flotantes amigables.
- **Client Components (`"use client"`)**: Se delega el mínimo indispensable al navegador (hooks pre-compilados y polling).

## 3. Componentes de Visualización (XAI & Drift)

### 3.1. ExplainabilityPanel
Visualiza las contribuciones SHAP utilizando gráficos de barras. Permite entender qué variables están influyendo más en una predicción específica, ya sea en tiempo real (Streaming) o en diferido.

### 3.2. DriftPanel
Integrado en la vista de Datasets, muestra la estabilidad de los datos. Permite ajustar los umbrales PSI y KS directamente desde la UI y visualizar qué columnas están sufriendo variaciones estadísticas significativas.

## 4. Estado Asíncrono de Tareas de Celery y WebSockets

1. **Polling (Celery)**: Para tareas de largo recorrido (Training, Batch Predictions, Entrenamiento NLP). El frontend consulta el estado cada 3s hasta llegar a un estado final.
2. **WebSockets (Streaming)**: Para inferencia inmediata. Se mantiene una conexión abierta que permite enviar filas de datos y recibir predicciones al instante sin la sobrecarga de HTTP.
3. **SSE (Streaming LLM)**: El asistente usa `text/event-stream` leído con `ReadableStream` para mostrar la respuesta token a token.

## 5. Sección NLP (Fase 1)

El tab **NLP** de la Sidebar (`Sparkles`) contiene 4 sub-tabs gestionados por `useNLP` y el contenedor `NLPTab.tsx`:

| Sub-tab | Componente | Funcionalidad |
|---------|-----------|---------------|
| **Training** | `NLPTrainingTab.tsx` | Selección de dataset + `text_column`/`label_column` + arquitectura (`GET /nlp/algorithms`) + tarea + estado del job (polling 3s). |
| **Inference** | `NLPInferenceTab.tsx` | Modelo NLP (filtro por `domain:"nlp"` en `useModels`) + textarea → label/confianza/entidades/tokens/latencia. Batch CSV con `POST /nlp/predict/batch`. |
| **Semantic Search** | `SemanticSearchTab.tsx` | Búsqueda semántica sobre modelos/datasets (`GET /nlp/search`) con score y metadata. |
| **Assistant** | `AssistantTab.tsx` | Chat con streaming SSE (`POST /nlp/assistant/chat`), fuentes citadas, historial y borrado. |

Tipos compartidos en `src/lib/api.ts` (`NLPAlgorithm`, `NLPPrediction`, `SemanticSearchResult`, `ChatResponse`, `ColumnTypesAnalysis`, etc.).

## 6. Evolución UX/UI y Nuevas Vistas (Fase 2)

El frontend ha sido unificado bajo una estética oscura de ingeniería de alto contraste (`#0b0c0f` / `#101217`, bordes `zinc-800` de `1px`, tipografía monoespaciada `tabular-nums`, CTAs monocromos y acento naranja `#f97316` puntual).

### Navegación y Productividad
- **Sincronización con URL (`?view=...`)**: `DashboardMVP.tsx` sincroniza la pestaña activa con la query string y soporta historial del navegador (`popstate`), evitando perder la vista al recargar.
- **Command Palette (`Cmd+K` / `Ctrl+K`)**: componente `CommandPalette.tsx` para saltar instantáneamente con teclado entre vistas, datasets, acciones rápidas o desplegar cualquier modelo.

### Nuevas Vistas y Componentes Clave
| Vista / Componente | Archivo | Funcionalidad |
|---|---|---|
| **Workspace Overview** | `WorkspaceOverviewTab.tsx` | Resumen ejecutivo de datasets, modelos, predicciones y plugins activos. |
| **Model Endpoints** | `ServingEndpointsTab.tsx` | Despliegue en 1 clic de endpoints dedicados, tráfico Canary (`0–100%`), consola JSON en vivo, snippet `curl` y visor de **Recent Invocation Logs**. |
| **Environments & Code** | `EnvironmentsPipelinesTab.tsx` | Gestión de `RuntimeEnvironments` (paquetes PyPI/uv, imagen base), editor de pipelines Python `def transform(df)` y panel de **Pluggable Backends** (7 registros). |
| **API Keys & Audit** | `GovernanceTab.tsx` | Creación/revocación de PATs (`px_...`), gestión de Workspaces, Audit Log y stream en vivo de eventos Redis Streams. |
| **Side-by-Side Compare** | `ModelCompareModal.tsx` | Comparador de 2–4 modelos en `ModelsTab` con resaltado de mejor métrica MLflow, hiperparámetros y acciones directas (*Deploy* / *Promote*). |
| **Preprocessing Builder** | `PipelineBuilder.tsx` | Botón **Auto-suggest**, chips clicables con tipo/nulos y editor interactivo para renombrar cualquier columna de salida. |
| **Plugins & Tools** | `CopilotStudioTab.tsx` | Inspección de plugins de dominio activos y catálogo de herramientas del Copilot gobernadas por RBAC. |
