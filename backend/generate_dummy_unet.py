import torch
from pathlib import Path
import sys

# Ensure backend acts as root
backend_path = Path(__file__).resolve().parent
sys.path.append(str(backend_path))

from app.core_ml.models.unet import UNet

def generate_unet():
    print("Instance UNet...")
    model = UNet(in_channels=3, num_classes=2)
    
    out_path = backend_path / "dummy_unet.pth"
    torch.save(model.state_dict(), out_path)
    print(f"Saved dummy UNet state_dict to: {out_path}")

if __name__ == "__main__":
    generate_unet()
