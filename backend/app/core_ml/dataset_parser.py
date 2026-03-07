import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ConfigurableDataset(Dataset):
    """
    Dataset Genérico y Seguro que interpreta un archivo config.json subido por el usuario
    junto a sus imágenes, en lugar de ejecutar scripts Python en texto plano.
    """
    def __init__(self, root_dir: str, config_path: str, is_train: bool = False):
        self.root_dir = root_dir
        self.config_path = config_path
        self.is_train = is_train

        # Leer JSON Configuration
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Missing config file at: {self.config_path}")

        with open(self.config_path, "r") as f:
            self.config = json.load(f)

        # Parse transforms
        self.transform = self._build_transforms()

        # Buscar imágenes
        img_folder = self.config.get("image_folder", "images")
        mask_folder = self.config.get("mask_folder", "masks")

        self.images_path = os.path.join(self.root_dir, img_folder)
        self.masks_path = os.path.join(self.root_dir, mask_folder)

        if not os.path.isdir(self.images_path):
            raise ValueError(f"Image folder '{img_folder}' not found in {self.root_dir}")

        self.image_files = sorted([
            f for f in os.listdir(self.images_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
        ])

    def _build_transforms(self):
        """Construye transformaciones estáticas dictadas por el JSON"""
        transform_ops = []
        tf_config = self.config.get("transforms", {})

        # 1. Resize
        target_size = tf_config.get("resize") # e.g. [256, 256]
        if target_size:
            transform_ops.append(transforms.Resize(tuple(target_size)))

        # 2. ToTensor
        transform_ops.append(transforms.ToTensor())

        # 3. Normalize
        norm_config = tf_config.get("normalize")
        if norm_config:
            mean = norm_config.get("mean", [0.485, 0.456, 0.406])
            std = norm_config.get("std", [0.229, 0.224, 0.225])
            transform_ops.append(transforms.Normalize(mean=mean, std=std))

        return transforms.Compose(transform_ops)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_path, img_name)

        # Load image via PIL
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Optional: Load Masks for training/evaluation
        mask = torch.zeros((1, *image.shape[1:])) # Default dummy mask
        if os.path.isdir(self.masks_path):
            mask_name = img_name # Asume mismo nombre (configurable en un futuro)
            mask_path = os.path.join(self.masks_path, mask_name)
            if os.path.exists(mask_path):
                mask_pil = Image.open(mask_path).convert("L")

                # Apply resize to mask if needed
                target_size = self.config.get("transforms", {}).get("resize")
                if target_size:
                    mask_pil = mask_pil.resize(tuple(target_size), Image.NEAREST)

                mask_t = transforms.ToTensor()(mask_pil)
                # Ensure mask is long tensor for 1-hot / categorical
                mask = (mask_t > 0).long()

        return image, mask
