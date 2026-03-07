"""
migrate.py — Script de migración manual.

Uso:
    cd backend
    .venv\\Scripts\\python.exe migrate.py

Añade la columna mlflow_inference_run_id a la tabla prediction si no existe,
y luego sincroniza cualquier tabla nueva que aún no exista en la BD
(equivalente a Base.metadata.create_all con checkfirst=True).
"""
import sys
from pathlib import Path

# Asegurar que el package `app` está en el path
sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy import inspect, text
from app.database import engine
import app.models  # noqa: F401  — registra todos los modelos en Base.metadata
from app.models.base import Base


def column_exists(table_name: str, column_name: str) -> bool:
    inspector = inspect(engine)
    columns = [col["name"] for col in inspector.get_columns(table_name)]
    return column_name in columns


def table_exists(table_name: str) -> bool:
    inspector = inspect(engine)
    return table_name in inspector.get_table_names()


def run_migrations():
    print("=== Iniciando migraciones ===")

    with engine.begin() as conn:
        if table_exists("prediction") and not column_exists("prediction", "mlflow_inference_run_id"):
            print("  [+] Añadiendo columna prediction.mlflow_inference_run_id ...")
            conn.execute(text(
                "ALTER TABLE prediction ADD COLUMN mlflow_inference_run_id VARCHAR;"
            ))
            print("  [OK] Columna añadida.")
        else:
            print("  [=] prediction.mlflow_inference_run_id ya existe o la tabla no existe aún.")
            
        # Modificar dataset_id para que sea nullable en inferencias de una imagen
        if table_exists("prediction"):
            print("  [+] Haciendo prediction.dataset_id nullable ...")
            conn.execute(text(
                "ALTER TABLE prediction ALTER COLUMN dataset_id DROP NOT NULL;"
            ))
            print("  [OK] Columna modificada.")

        # Añadir input_image_path a prediction si no existe
        if table_exists("prediction") and not column_exists("prediction", "input_image_path"):
            print("  [+] Añadiendo columna prediction.input_image_path ...")
            conn.execute(text(
                "ALTER TABLE prediction ADD COLUMN input_image_path VARCHAR;"
            ))
            print("  [OK] Columna añadida.")
        else:
            print("  [=] prediction.input_image_path ya existe o la tabla no existe aún.")

        # Añadir created_at a ml_model si no existe
        if table_exists("ml_model") and not column_exists("ml_model", "created_at"):
            print("  [+] Añadiendo columna ml_model.created_at ...")
            conn.execute(text(
                "ALTER TABLE ml_model ADD COLUMN created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP;"
            ))
            print("  [OK] Columna añadida.")
        else:
            print("  [=] ml_model.created_at ya existe o la tabla no existe aún.")

        # Añadir preprocessing_pipeline_path a ml_model si no existe
        if table_exists("ml_model") and not column_exists("ml_model", "preprocessing_pipeline_path"):
            print("  [+] Añadiendo columna ml_model.preprocessing_pipeline_path ...")
            conn.execute(text(
                "ALTER TABLE ml_model ADD COLUMN preprocessing_pipeline_path VARCHAR;"
            ))
            print("  [OK] Columna añadida.")
        else:
            print("  [=] ml_model.preprocessing_pipeline_path ya existe o la tabla no existe aún.")

        # ── Fase 1: Data Ops — nuevas columnas en dataset ────────────────────
        if table_exists("dataset"):
            new_dataset_cols = {
                "file_type": "VARCHAR",
                "num_rows": "INTEGER",
                "num_columns": "INTEGER",
                "column_names": "JSON",
                "version": "INTEGER DEFAULT 1",
                "mlflow_artifact_uri": "VARCHAR",
                "pipeline_path": "VARCHAR",
            }
            for col_name, col_type in new_dataset_cols.items():
                if not column_exists("dataset", col_name):
                    print(f"  [+] Añadiendo columna dataset.{col_name} ...")
                    conn.execute(text(
                        f"ALTER TABLE dataset ADD COLUMN {col_name} {col_type};"
                    ))
                    print(f"  [OK] Columna dataset.{col_name} añadida.")
                else:
                    print(f"  [=] dataset.{col_name} ya existe.")

        # ── Fase 0: Seguridad — RBAC: columna role en users ─────────────────
        if table_exists("users") and not column_exists("users", "role"):
            print("  [+] Añadiendo columna users.role (RBAC) ...")
            conn.execute(text(
                "ALTER TABLE users ADD COLUMN role VARCHAR NOT NULL DEFAULT 'viewer';"
            ))
            print("  [OK] Columna users.role añadida (default: 'viewer').")
        else:
            print("  [=] users.role ya existe o la tabla users no existe aún.")

        # ── Fase 0: Seguridad — created_at en users ───────────────────────────
        if table_exists("users") and not column_exists("users", "created_at"):
            print("  [+] Añadiendo columna users.created_at ...")
            conn.execute(text(
                "ALTER TABLE users ADD COLUMN created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP;"
            ))
            print("  [OK] Columna users.created_at añadida.")
        else:
            print("  [=] users.created_at ya existe o la tabla users no existe aún.")

        # ── Fase 0: Seguridad — Quota Limiting: columnas en tenant ──────────
        if table_exists("tenant"):
            quota_cols = {
                "max_datasets": "INTEGER DEFAULT 50",
                "max_models": "INTEGER DEFAULT 20",
                "max_predictions_per_day": "INTEGER DEFAULT 500",
                "max_training_jobs_per_day": "INTEGER DEFAULT 10",
            }
            for col_name, col_type in quota_cols.items():
                if not column_exists("tenant", col_name):
                    print(f"  [+] Añadiendo columna tenant.{col_name} ...")
                    conn.execute(text(
                        f"ALTER TABLE tenant ADD COLUMN {col_name} {col_type};"
                    ))
                    print(f"  [OK] Columna tenant.{col_name} añadida.")
                else:
                    print(f"  [=] tenant.{col_name} ya existe.")

    # 2. Crear cualquier tabla nueva que aún no esté en la BD
    print("  [+] Sincronizando tablas (create_all checkfirst=True) ...")
    Base.metadata.create_all(bind=engine, checkfirst=True)
    print("  [OK] Tablas sincronizadas.")

    # 3. Seed default tenant if it doesn't exist
    DEFAULT_TENANT_ID = "00000000-0000-0000-0000-000000000000"
    from app.models.tenant import Tenant
    from sqlalchemy.orm import Session

    with Session(engine) as session:
        tenant = session.query(Tenant).filter(Tenant.id == DEFAULT_TENANT_ID).first()
        if not tenant:
            print(f"  [+] Sembrando tenant por defecto (ID: {DEFAULT_TENANT_ID}) ...")
            default_tenant = Tenant(
                id=DEFAULT_TENANT_ID,
                name="Default Tenant",
                is_active=True
            )
            session.add(default_tenant)
            session.commit()
            print("  [OK] Tenant por defecto creado.")
        else:
            print(f"  [=] Tenant por defecto (ID: {DEFAULT_TENANT_ID}) ya existe.")

    print("=== Migraciones completadas ===")


if __name__ == "__main__":
    run_migrations()
