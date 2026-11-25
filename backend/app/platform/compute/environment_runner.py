"""
EnvironmentRunner & EnvironmentRunnerRegistry — Ejecución aislada de código y
pipelines definidos por el usuario en entornos con dependencias propias.

Backends incluidos:
  - `uv_isolated`: crea/reutiliza un entorno efímero aislado con `uv run --with ...`
    (o el intérprete actual si no se solicitan paquetes extra) para ejecutar una
    función `transform(df)` sobre un DataFrame tabular.
Extensible mediante `register_environment_runner(name, factory)` a Docker,
Kubernetes Jobs o Ray runtime_env sin tocar los routers ni los pipelines.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import pandas as pd

from app.platform.compute.base import EnvironmentSpec

logger = logging.getLogger(__name__)

_HARNESS_TEMPLATE = '''
import sys
import pandas as pd

in_path = sys.argv[1]
out_path = sys.argv[2]

df = pd.read_csv(in_path)

# ── Código definido por el usuario ──
{user_code}

if "transform" not in globals() or not callable(globals()["transform"]):
    raise ValueError("El código del pipeline debe definir una función `def transform(df): ...`")

result = transform(df)
if not isinstance(result, pd.DataFrame):
    raise ValueError("`transform(df)` debe retornar un pandas.DataFrame.")

result.to_csv(out_path, index=False)
'''


@dataclass
class ExecutionResult:
    dataframe: pd.DataFrame
    stdout: str
    stderr: str
    runner: str


class EnvironmentRunner(ABC):
    """Contrato para ejecutar transformaciones de usuario dentro de un EnvironmentSpec."""

    runner_name: str

    @abstractmethod
    def run_dataframe_transform(
        self,
        df: pd.DataFrame,
        user_code: str,
        env_spec: EnvironmentSpec,
        timeout_seconds: int = 120,
    ) -> ExecutionResult:
        ...


class UvIsolatedRunner(EnvironmentRunner):
    """
    Ejecuta código Python definido por el usuario en un subproceso aislado.
    Cuando el entorno declara `packages` adicionales, invoca `uv run --no-project
    --with <pkg> ...` para resolver esas librerías de forma aislada; si no, usa
    el entorno Python activo del worker.
    """

    runner_name = "uv_isolated"

    def _build_command(self, script_path: str, in_path: str, out_path: str, env_spec: EnvironmentSpec) -> List[str]:
        packages = [p.strip() for p in (env_spec.packages or []) if p and p.strip()]
        uv_bin = shutil.which("uv")
        if packages and uv_bin:
            cmd = [uv_bin, "run", "--no-project"]
            # Garantizar pandas para la interfaz DataFrame del harness
            base_pkgs = list(packages)
            if not any(p.lower().startswith("pandas") for p in base_pkgs):
                base_pkgs.insert(0, "pandas")
            for pkg in base_pkgs:
                cmd.extend(["--with", pkg])
            cmd.extend(["python", script_path, in_path, out_path])
            return cmd

        return [sys.executable, script_path, in_path, out_path]

    def run_dataframe_transform(
        self,
        df: pd.DataFrame,
        user_code: str,
        env_spec: EnvironmentSpec,
        timeout_seconds: int = 120,
    ) -> ExecutionResult:
        if not user_code or not user_code.strip():
            raise ValueError("`custom_code` no puede estar vacío.")

        with tempfile.TemporaryDirectory(prefix="praxisml_env_") as tmpdir:
            tmp = Path(tmpdir)
            in_path = tmp / "input.csv"
            out_path = tmp / "output.csv"
            script_path = tmp / "runner_script.py"

            df.to_csv(in_path, index=False)
            script_path.write_text(
                _HARNESS_TEMPLATE.format(user_code=user_code),
                encoding="utf-8",
            )

            cmd = self._build_command(str(script_path), str(in_path), str(out_path), env_spec)
            proc_env = {
                "PATH": os.environ.get("PATH", ""),
                "HOME": os.environ.get("HOME", str(tmp)),
                "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR", str(Path(tempfile.gettempdir()) / "uv-cache")),
                **{str(k): str(v) for k, v in (env_spec.env_vars or {}).items()},
            }

            try:
                completed = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    env=proc_env,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"La ejecución del pipeline excedió el timeout de {timeout_seconds}s."
                ) from exc

            if completed.returncode != 0:
                err_msg = (completed.stderr or completed.stdout or "Error desconocido").strip()
                raise ValueError(f"Error ejecutando código del pipeline: {err_msg[-800:]}")

            if not out_path.exists():
                raise ValueError("El script no generó ningún DataFrame de salida.")

            out_df = pd.read_csv(out_path)
            return ExecutionResult(
                dataframe=out_df,
                stdout=completed.stdout[-2000:],
                stderr=completed.stderr[-2000:],
                runner=self.runner_name,
            )


class DockerEnvironmentRunner(EnvironmentRunner):
    """
    Ejecuta código de usuario en un contenedor Docker efímero sin red, con
    límites estrictos de CPU y memoria usando la `base_image` del entorno.
    """

    runner_name = "docker"

    def run_dataframe_transform(
        self,
        df: pd.DataFrame,
        user_code: str,
        env_spec: EnvironmentSpec,
        timeout_seconds: int = 180,
    ) -> ExecutionResult:
        docker_bin = shutil.which("docker")
        if not docker_bin:
            raise RuntimeError("CLI de Docker no disponible en este host.")
        if not user_code or not user_code.strip():
            raise ValueError("`custom_code` no puede estar vacío.")

        with tempfile.TemporaryDirectory(prefix="praxisml_docker_") as tmpdir:
            tmp = Path(tmpdir)
            in_path = tmp / "input.csv"
            out_path = tmp / "output.csv"
            script_path = tmp / "runner_script.py"

            df.to_csv(in_path, index=False)
            script_path.write_text(
                _HARNESS_TEMPLATE.format(user_code=user_code),
                encoding="utf-8",
            )

            image = env_spec.base_image or "python:3.10-slim"
            cmd = [
                docker_bin,
                "run",
                "--rm",
                "--network=none",
                "--memory=512m",
                "--cpus=1",
                "-v",
                f"{tmp}:/work",
            ]
            for k, v in (env_spec.env_vars or {}).items():
                cmd.extend(["-e", f"{k}={v}"])
            cmd.extend([image, "python", "/work/runner_script.py", "/work/input.csv", "/work/output.csv"])

            try:
                completed = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"Contenedor Docker excedió el timeout de {timeout_seconds}s."
                ) from exc

            if completed.returncode != 0:
                err_msg = (completed.stderr or completed.stdout or "Error en contenedor Docker").strip()
                raise ValueError(f"Error ejecutando en contenedor Docker: {err_msg[-800:]}")

            if not out_path.exists():
                raise ValueError("El contenedor no generó ningún DataFrame de salida.")

            return ExecutionResult(
                dataframe=pd.read_csv(out_path),
                stdout=completed.stdout[-2000:],
                stderr=completed.stderr[-2000:],
                runner=self.runner_name,
            )


class EnvironmentRunnerRegistry:
    """Registro pluggable de ejecutores de entornos (uv_isolated, docker, k8s, ray)."""

    def __init__(self) -> None:
        self._factories: Dict[str, Callable[[], EnvironmentRunner]] = {
            "uv_isolated": UvIsolatedRunner,
            "docker": DockerEnvironmentRunner,
        }

    def register(self, name: str, factory: Callable[[], EnvironmentRunner]) -> None:
        self._factories[name.lower()] = factory

    def list_runners(self) -> List[str]:
        return sorted(self._factories.keys())

    def create(self, name: str) -> EnvironmentRunner:
        key = (name or "uv_isolated").lower()
        factory = self._factories.get(key)
        if factory is None:
            raise NotImplementedError(
                f"EnvironmentRunner '{name}' no registrado (disponibles: {', '.join(self.list_runners())})."
            )
        return factory()


_runner_registry = EnvironmentRunnerRegistry()


def register_environment_runner(
    name: str, factory: Callable[[], EnvironmentRunner]
) -> None:
    _runner_registry.register(name, factory)


def get_environment_runner_registry() -> EnvironmentRunnerRegistry:
    return _runner_registry
