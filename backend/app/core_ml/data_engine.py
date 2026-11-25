"""
DataEngine & DataEngineRegistry — Abstracción de motor de datos tabular.

Permite operar con `pandas` por defecto y enchufar motores out-of-core o
distribuidos (`polars`, `duckdb`, `spark`, `ray_data`) mediante
`register_data_engine(name, factory)` sin reescribir parsers ni routers.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import pandas as pd


class DataEngine(ABC):
    """Contrato común para lectura y escritura de datos tabulares."""

    engine_name: str

    @abstractmethod
    def read(
        self,
        source: Union[str, Path, bytes, BytesIO],
        file_type: str,
    ) -> pd.DataFrame:
        ...

    @abstractmethod
    def write_bytes(self, df: pd.DataFrame, file_type: str = "csv") -> bytes:
        ...


class PandasDataEngine(DataEngine):
    engine_name = "pandas"

    def read(
        self,
        source: Union[str, Path, bytes, BytesIO],
        file_type: str,
    ) -> pd.DataFrame:
        if isinstance(source, bytes):
            source = BytesIO(source)
        ft = file_type.lower().lstrip(".")
        if ft == "csv":
            return pd.read_csv(source)
        if ft in ("xlsx", "xls", "excel"):
            return pd.read_excel(source, engine="openpyxl")
        if ft == "parquet":
            return pd.read_parquet(source, engine="pyarrow")
        raise ValueError(f"Formato tabular no soportado: {file_type}")

    def write_bytes(self, df: pd.DataFrame, file_type: str = "csv") -> bytes:
        buf = BytesIO()
        ft = file_type.lower().lstrip(".")
        if ft == "csv":
            df.to_csv(buf, index=False)
            return buf.getvalue()
        if ft == "parquet":
            df.to_parquet(buf, index=False)
            return buf.getvalue()
        raise ValueError(f"Escritura tabular no soportada para formato: {file_type}")


class DataEngineRegistry:
    """Registro pluggable de motores de procesamiento de datos tabulares."""

    def __init__(self) -> None:
        self._factories: Dict[str, Callable[[], DataEngine]] = {
            "pandas": PandasDataEngine,
        }

    def register(self, name: str, factory: Callable[[], DataEngine]) -> None:
        self._factories[name.lower()] = factory

    def list_engines(self) -> List[str]:
        return sorted(self._factories.keys())

    def create(self, name: str) -> DataEngine:
        key = name.lower()
        factory = self._factories.get(key)
        if factory is None:
            raise ValueError(
                f"DataEngine '{name}' no registrado (disponibles: {', '.join(self.list_engines())})."
            )
        return factory()


_data_engine_registry = DataEngineRegistry()


def register_data_engine(name: str, factory: Callable[[], DataEngine]) -> None:
    _data_engine_registry.register(name, factory)


def get_data_engine_registry() -> DataEngineRegistry:
    return _data_engine_registry


def get_data_engine(engine_name: Optional[str] = None) -> DataEngine:
    selected = (engine_name or os.getenv("DATA_ENGINE", "pandas")).lower()
    return _data_engine_registry.create(selected)
