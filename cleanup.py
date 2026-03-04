"""
cleanup.py — Elimina los scripts de investigación que ya están incorporados al backend.
Ejecutar DESDE la raíz del proyecto:

    python cleanup.py
"""
import os
from pathlib import Path

ROOT = Path(__file__).parent

SURPLUS_FILES = [
    # Scripts de pipeline de investigación (reemplazados por el backend)
    "combined_pipeline.py",
    "combined_pipeline_medsam.py",
    "combined_pipeline_own_model.py",
    "combined_pipeline_universeg.py",
    "combined_pipeline_universeg_mean.py",
    "combined_pipeline_universeg_unique_channel.py",
    # Algoritmos de incertidumbre standalone (ahora en backend/app/core_ml/)
    "mc_dropout.py",
    "tta.py",
    "noise_inference.py",
    # Scripts de calibración escalada (investigación)
    "scaled_mc_dropout_cross_entropy.py",
    "scaled_mc_dropout_cross_entropy_bisection.py",
    "scaled_mc_dropout_hamming.py",
    "scaled_mc_dropout_hamming_bisection.py",
    "scaled_mc_dropout_nll_relaxed.py",
    "scaled_mc_dropout_nll_relaxed_bisection.py",
    "compare_scaled_methods.py",
    "compare_scaled_metrics_mri.py",
    # Scripts de entrenamiento / evaluación standalone
    "train.py",
    "evaluate.py",
    "evaluate_mc.py",
    "test_uncertainty.py",
    "test_unet.py",
    "prepare_carvana_dataset.py",
    # Configs YAML de pipelines viejos
    "combined_config.yaml",
    "combined_config_medsam.yaml",
    "combined_config_universeg.yaml",
    # Shell scripts de ejecución en GPU
    "execute_gpu_medsam.sh",
    "execute_gpu_universeg.sh",
    "execute_gpu_universeg_mean.sh",
    "execute_gpu_universeg_unique.sh",
    "execution_pipeline.sh",
    # Otros sobrantes
    "car.png",
    "requirements.txt",   # Reemplazado por backend/pyproject.toml
]

if __name__ == "__main__":
    deleted, skipped = 0, 0
    for fname in SURPLUS_FILES:
        fpath = ROOT / fname
        if fpath.exists():
            fpath.unlink()
            print(f"  [✓] Eliminado: {fname}")
            deleted += 1
        else:
            print(f"  [=] No encontrado (ya eliminado?): {fname}")
            skipped += 1
    print(f"\nResumen: {deleted} eliminados, {skipped} no encontrados.")
