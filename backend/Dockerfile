FROM python:3.10-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Definimos dónde vivirá el entorno virtual (FUERA de /app)
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
# Aseguramos que el PATH apunte allí
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copiamos solo los archivos de dependencias primero para aprovechar la caché
COPY pyproject.toml uv.lock ./

# Instalamos las dependencias. 
# Nota: Quitamos --path y dejamos que use UV_PROJECT_ENVIRONMENT
RUN uv sync --frozen --no-dev --no-install-project

# Ahora copiamos el resto del código
COPY . .

EXPOSE 8000

# Al estar en /opt/venv, el uvicorn instalado allí siempre será visible
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]