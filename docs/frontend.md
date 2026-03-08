# Frontend Documentation (Next.js)

El cliente principal de PraxisML, una SPA (Single Page Application) servida desde servidor, construida con los últimos estándares del ecosistema React.

## 1. Stack Tecnológico
- **Framework Web**: `Next.js 16` (App Router `src/app/`).
- **Librería UI**: `React 19`.
- **Ecosistema CSS**: `TailwindCSS v4` completo implementando `postcss`.
- **Componentes Base**: `Radix UI` proporcionando accesibilidad y modales primitivos limpios (e.g. `lucide-react` para iconos gráficos).
- **Tipado**: `TypeScript 5` estricto de end-to-end con `eslint-config-next`.
- **Notificaciones/Alertas**: `react-hot-toast` para propagación de errores provenientes de tareas asíncronas de Celery, con visualización atractiva de validaciones JWT o RBAC de forma unificada.
- **Charts y Gráficos**: `recharts` para visualizaciones, paneles del tenant y resúmenes de *Profiling* o Métricas de MLflow en frontend (si aplicase implementarlo así o en dashboards nativos).

## 2. Estilo de Código y Directrices Generales

- **App Router (`src/app`)**: Navegación principal, SSR o CSR, Layouts anidados por roles. Existen páginas estáticas para pre-login (como la web promocional y la pantalla de autentiación / registro). Todo requiere un JWT válido excepto *Login / Sign Up*.
- **Auth Provider**: El patrón context encapsulado, persistiendo o refrescando el token de `FastAPI` transparentemente (OAuth2). Si expira el token y lanza 401 el Backend, Next renderiza el Modal reautorizador y enruta de vuelta hacia `login`.
- **Drag & Drop**: Funciones construidas nativamente o con utilidades React para la carga local de Datasets (`ZIP`, `.json`) y modelos seguros de Pytorch (`.pt`) directamente a los endpoints `/upload` del Backend.
- **Manejo de Errores de API (4xx / 5xx)**: Componente axios o interceptor global de `fetch` intercepta un 403 (Permisos Denegados, ej. ser `VIEWER` intentando subir modelo), un 429 (Cuota Consumida `QuotaExceededError`) u otra variante, reflejándolos de vuelta en alertas flotantes amigables y cancelando el evento.
- **Client Components (`"use client"`)**: Se delega el mínimo indispensable al navegador (hooks pre-compilados y polling).

## 3. Estado Asíncrono de Tareas de Celery y Polling

Los flujos vitales de Machine Learning (entrenamiento o procesamiento pesado) no bloquean el servidor ni bloquean el cliente:
1. El usuario interactúa: ej. "Entrenar".
2. Next.js hace `POST` al backend.
3. Backend devuelve un `HTTP 202 Accepted` y JSON `{ task_id: "xyz" ... }`.
4. El Frontend Next.js suscribe un ciclo de *Polling*: `interval` (cada 2 segundos, `GET /api/v1/training/status/xyz`).
5. Cuando el status finaliza en `SUCCESS` o `FAILURE`, se limpia el `interval` y se pinta por renderizado condicional el resultado final o gráfica, con un *toast success*.
6. La UX mantiene una visualización estilo barra de carga o "Pendiente" unificada globalmente para no pausar a los creadores (editors) mientras el Worker de Redis resuelve el entrenamiento en *the background*.

## 4. UI/UX: Modos Claros/Oscuros y Componentes Base

El proyecto soporta modales nativos Tailwind CSS utilizando `clsx` y `tailwind-merge` (`twMerge`) para fusionar clases estables con inyectadas. Aportando animaciones a los botones (e.g `class-variance-authority`), asegurando uniformidad transversal (botones, badges para roles de equipo de trabajo, etc).
