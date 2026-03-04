"""
Utilidades para parsear archivos tabulares (.csv, .xlsx, .parquet).
Extrae metadata (schema, filas, columnas) y ofrece preview.
"""
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Formatos soportados y su extensión
TABULAR_EXTENSIONS = {".csv", ".xlsx", ".parquet"}
IMAGE_EXTENSIONS = {".zip"}
ALL_SUPPORTED = TABULAR_EXTENSIONS | IMAGE_EXTENSIONS


def detect_file_type(filename: str) -> Optional[str]:
    """Detecta el tipo de archivo por su extensión."""
    ext = Path(filename).suffix.lower()
    if ext in TABULAR_EXTENSIONS:
        return ext.lstrip(".")
    if ext in IMAGE_EXTENSIONS:
        return ext.lstrip(".")
    return None


def is_tabular(file_type: str) -> bool:
    """Devuelve True si el tipo de archivo es tabular."""
    return file_type in ("csv", "xlsx", "parquet")


def read_tabular(file_path: str, file_type: Optional[str] = None) -> pd.DataFrame:
    """
    Lee un archivo tabular y devuelve un DataFrame de pandas.

    Args:
        file_path: Ruta absoluta al archivo.
        file_type: Tipo de archivo (csv, xlsx, parquet). Si None, se detecta.

    Returns:
        pd.DataFrame con los datos.

    Raises:
        ValueError: si el formato no es soportado.
        FileNotFoundError: si el archivo no existe.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Archivo no encontrado: {file_path}")

    if file_type is None:
        file_type = detect_file_type(file_path)

    if file_type == "csv":
        return pd.read_csv(file_path)
    elif file_type == "xlsx":
        return pd.read_excel(file_path, engine="openpyxl")
    elif file_type == "parquet":
        return pd.read_parquet(file_path, engine="pyarrow")
    else:
        raise ValueError(f"Formato tabular no soportado: {file_type}")


def extract_metadata(file_path: str, file_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Extrae metadata de un archivo tabular sin cargar todo en memoria
    (excepto para contar filas, que requiere lectura completa).

    Returns:
        dict con claves: num_rows, num_columns, column_names, column_dtypes
    """
    df = read_tabular(file_path, file_type)
    return {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "column_dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }


def get_preview(
    file_path: str,
    file_type: Optional[str] = None,
    max_rows: int = 20,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Lee un archivo tabular y devuelve un preview (primeras N filas) junto a su metadata.

    Returns:
        (preview_df, metadata_dict)
    """
    df = read_tabular(file_path, file_type)
    metadata = {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "column_dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }
    preview_df = df.head(max_rows)
    return preview_df, metadata


def validate_schema_match(
    train_path: str,
    test_path: str,
    train_type: Optional[str] = None,
    test_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compara el esquema de dos datasets tabulares.
    Útil para validación de esquema (Extras del PM).

    Returns:
        dict con claves: match (bool), missing_columns, extra_columns
    """
    train_df = read_tabular(train_path, train_type)
    test_df = read_tabular(test_path, test_type)

    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)

    missing = train_cols - test_cols
    extra = test_cols - train_cols

    return {
        "match": len(missing) == 0 and len(extra) == 0,
        "missing_columns": list(missing),
        "extra_columns": list(extra),
    }
