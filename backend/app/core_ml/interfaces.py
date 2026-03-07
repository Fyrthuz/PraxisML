from abc import ABC, abstractmethod
import numpy as np
from typing import Dict

class IUncertaintyAlgorithm(ABC):
    """
    Interfaz unificada para encapsular la lógica de los algoritmos de incertidumbre.
    Actúa como puente entre la API web (datos crudos/NumPy) y el tensor de PyTorch inferido.
    """

    @abstractmethod
    def estimate_uncertainty(self, input_data: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """
        Ejecuta el algoritmo de incertidumbre en el array de entrada.

        Args:
            input_data (np.ndarray): Imagen o lote de imágenes procesadas.
            **kwargs: Parámetros adicionales para la inferencia.

        Returns:
            Dict[str, np.ndarray]: Diccionario con las claves 'prediction' y 'uncertainty',
            listo para ser serializado, guardado en disco o expuesto desde el endpoint de FastAPI.
        """
        pass
