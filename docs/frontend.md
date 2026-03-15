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

- **App Router (`src/app`)**: Navegación principal. Incluye secciones para `Datasets`, `Models`, `Predictions`, y la nueva sección de `Streaming`.
- **Auth Provider**: El patrón context encapsulado, persistiendo o refrescando el token de `FastAPI` transparentemente (OAuth2).
- **Custom Hooks (`src/hooks`)**: Lógica encapsulada para funcionalidades complejas:
    - `useDrift`: Gestión de reportes de drift y actualización de umbrales.
    - `useStreamingInference`: Manejo del ciclo de vida de WebSockets para inferencia en tiempo real.
    - `usePredictions`: Polling y resultados de inferencia batch/single.
- **Drag & Drop**: Funciones para la carga local de Datasets (`ZIP`, `.json`) y modelos seguros de Pytorch (`.pt`).
- **Manejo de Errores de API (4xx / 5xx)**: Interceptor global que refleja errores en alertas flotantes amigables.
- **Client Components (`"use client"`)**: Se delega el mínimo indispensable al navegador (hooks pre-compilados y polling).

## 3. Componentes de Visualización (XAI & Drift)

### 3.1. ExplainabilityPanel
Visualiza las contribuciones SHAP utilizando gráficos de barras. Permite entender qué variables están influyendo más en una predicción específica, ya sea en tiempo real (Streaming) o en diferido.

### 3.2. DriftPanel
Integrado en la vista de Datasets, muestra la estabilidad de los datos. Permite ajustar los umbrales PSI y KS directamente desde la UI y visualizar qué columnas están sufriendo variaciones estadísticas significativas.

## 4. Estado Asíncrono de Tareas de Celery y WebSockets

1. **Polling (Celery)**: Para tareas de largo recorrido (Training, Batch Predictions). El frontend consulta el estado cada 2s hasta llegar a un estado final.
2. **WebSockets (Streaming)**: Para inferencia inmediata. Se mantiene una conexión abierta que permite enviar filas de datos y recibir predicciones al instante sin la sobrecarga de HTTP.

## 5. UI/UX: Modos Claros/Oscuros y Componentes Base

El proyecto soporta modales nativos Tailwind CSS utilizando `clsx` y `tailwind-merge` (`twMerge`) para fusionar clases estables con inyectadas. Aportando animaciones a los botones (e.g `class-variance-authority`), asegurando uniformidad transversal (botones, badges para roles de equipo de trabajo, etc).
