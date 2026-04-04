from typing import Dict, Type

import torch.nn as nn

from app.core_ml.models.unet import UNet


class ModelFactory:
    """
    Registry for model architectures to facilitate loading state_dicts.
    """
    _REGISTRY: Dict[str, Type[nn.Module]] = {
        "unet": UNet,
        # Add more architectures here as they are implemented
    }

    @classmethod
    def get_model(cls, architecture: str, **kwargs) -> nn.Module:
        """
        Instantiate a model based on the architecture name and parameters.
        """
        arch_lower = architecture.lower()
        if arch_lower not in cls._REGISTRY:
            # Fallback if architecture is not explicitly registered
            # but we might want to try to find it in the models directory?
            raise ValueError(f"Architecture '{architecture}' is not registered in ModelFactory.")

        model_class = cls._REGISTRY[arch_lower]
        return model_class(**kwargs)

    @classmethod
    def register(cls, name: str, model_class: Type[nn.Module]):
        cls._REGISTRY[name.lower()] = model_class
