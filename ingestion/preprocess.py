#this file reads images from data/product_eyewear/images/ and resizes them, coverting it to RGB. It also normalizes it and returns a pytorch tensor ready for ResNet which ca be used by offline embedding scripts


import io
import os
from PIL import Image
import torch
import torchvision.transforms as T

# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------

IMAGE_SIZE = 224
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------
# Preprocessing pipeline (ImageNet standard)
# ---------------------------------------------------------

_preprocess_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ---------------------------------------------------------
# Image loading utilities
# ---------------------------------------------------------

def load_image_from_path(path: str) -> Image.Image:
    """
    Load image from disk and convert to RGB.
    """
    return Image.open(path).convert("RGB")


def load_image_from_bytes(file_bytes: bytes) -> Image.Image:
    """
    Load image from raw bytes (Flask upload).
    """
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")

# ---------------------------------------------------------
# Core preprocessing
# ---------------------------------------------------------

def preprocess_image(img: Image.Image, device: str = "cpu") -> torch.Tensor:
    """
    Apply resize + normalization.
    
    Returns:
        Tensor of shape (1, 3, 224, 224)
    """
    tensor = _preprocess_transform(img)
    tensor = tensor.unsqueeze(0)  # add batch dimension
    return tensor.to(device)

# ---------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------

def preprocess_from_path(path: str, device: str = "cpu") -> torch.Tensor:
    """
    Path → preprocessed tensor
    """
    img = load_image_from_path(path)
    return preprocess_image(img, device=device)


def preprocess_from_bytes(file_bytes: bytes, device: str = "cpu") -> torch.Tensor:
    """
    Flask upload bytes → preprocessed tensor
    """
    img = load_image_from_bytes(file_bytes)
    return preprocess_image(img, device=device)

# ---------------------------------------------------------
# Batch preprocessing (dataset ingestion)
# ---------------------------------------------------------

def preprocess_directory(image_dir: str, device: str = "cpu"):
    """
    Preprocess all valid images in a directory.

    Returns:
        batch_tensor: Tensor of shape (N, 3, 224, 224)
        image_names: List of image filenames
    """
    tensors = []
    image_names = []

    for name in sorted(os.listdir(image_dir)):
        if not name.lower().endswith(IMAGE_EXTENSIONS):
            continue

        path = os.path.join(image_dir, name)

        try:
            img = load_image_from_path(path)
            tensor = _preprocess_transform(img)
            tensors.append(tensor)
            image_names.append(name)

        except Exception as e:
            print(f"[WARN] Skipping {name}: {e}")

    if not tensors:
        raise RuntimeError("No valid images found for preprocessing.")

    batch = torch.stack(tensors).to(device)
    return batch, image_names