"""
env.py — Configuración de Alembic para PraxisML.

Lee DATABASE_URL de la configuración de la app (app.core.config.settings)
para que no haya duplicación de credenciales.

Soporte para:
- Migrations offline (genera SQL sin conectar a la BD)
- Migrations online  (conecta directamente a PostgreSQL)
"""

import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import engine_from_config, pool

from alembic import context

# ── Asegurar que el package `app` está en sys.path ──────────────────────────
# env.py se ejecuta desde backend/, así que el package app ya debería estar
# accesible. Por si acaso, lo añadimos explícitamente.
sys.path.insert(0, str(Path(__file__).parents[1]))

# ── Cargar settings y modelos ────────────────────────────────────────────────
from app.core.config import settings          # noqa: E402
import app.models  # noqa: F401, E402  ← registra todos los modelos en Base.metadata
from app.models.base import Base             # noqa: E402

# ── Configuración Alembic ────────────────────────────────────────────────────
config = context.config

# Inyectar DATABASE_URL desde settings (sobrescribe el valor vacío de alembic.ini)
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

# Configurar logging si hay un fichero de configuración
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Metadatos de todos los modelos — Alembic usará esto para autogenerate
target_metadata = Base.metadata


# ── Modo offline ─────────────────────────────────────────────────────────────

def run_migrations_offline() -> None:
    """
    Genera SQL de migraciones sin conectar a la BD.
    Útil para revisar las migraciones antes de aplicarlas.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


# ── Modo online ──────────────────────────────────────────────────────────────

def run_migrations_online() -> None:
    """
    Aplica las migraciones conectando a la BD en tiempo real.
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()


# ── Punto de entrada ─────────────────────────────────────────────────────────

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
