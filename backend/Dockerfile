FROM python:3.10-slim AS base

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml uv.lock ./

# Perfil de instalación opcional:
#   all     -> instala todas las dependencias del lockfile
#   core    -> solo Control Plane ligero (sin torch/transformers/tabular)
#   tabular -> Control Plane + extra [tabular]
#   nlp     -> Control Plane + extras [pytorch, nlp]
#   cv      -> Control Plane + extras [pytorch, cv]
#   ts      -> Control Plane + extra [ts]
ARG PROFILE=all
RUN if [ "$PROFILE" = "core" ]; then \
        uv sync --frozen --no-dev --no-install-project; \
    elif [ "$PROFILE" = "tabular" ]; then \
        uv sync --frozen --no-dev --no-install-project --extra tabular --extra storage-s3; \
    elif [ "$PROFILE" = "nlp" ]; then \
        uv sync --frozen --no-dev --no-install-project --extra pytorch --extra nlp --extra storage-s3; \
    elif [ "$PROFILE" = "cv" ]; then \
        uv sync --frozen --no-dev --no-install-project --extra pytorch --extra cv --extra storage-s3; \
    elif [ "$PROFILE" = "ts" ]; then \
        uv sync --frozen --no-dev --no-install-project --extra ts --extra storage-s3; \
    else \
        uv sync --frozen --no-dev --no-install-project; \
    fi

COPY . .

EXPOSE 8000

CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
