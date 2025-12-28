# ingestion.preprocess
# --------------------
# Utilities to load and preprocess images for the embedding pipeline and
# for runtime uploads. This module provides small, well-documented helpers
# that convert images to RGB, resize them to the ImageNet standard size,
# normalize using ImageNet statistics, and return ready-to-run PyTorch
# Tensors for a ResNet50 backbone.
#
# Key functions:
# - `preprocess_from_path` — read image from disk, convert to tensor
# - `preprocess_from_bytes` — convert Flask-uploaded bytes to tensor
# - `preprocess_directory` — batch preprocess for dataset ingestion
#
# These helpers keep preprocessing consistent between offline embedding
# generation and the live Flask service.

import io
import os
from PIL import Image
import torch
import torchvision.transforms as T

# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------

IMAGE_SIZE = 224
# Accept common image file extensions used in the product dataset
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

# ImageNet normalization (used by pretrained torchvision models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------
# Preprocessing pipeline (ImageNet standard)
# ---------------------------------------------------------

_preprocess_transform = T.Compose([
    # Resize to the expected input size for ResNet (H, W)
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    # Convert PIL -> torch.FloatTensor and scale to [0, 1]
    T.ToTensor(),
    # Standard ImageNet normalization
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
    # Apply the composed transform (resize -> to-tensor -> normalize)
    tensor = _preprocess_transform(img)

    # Add a batch dimension since the embedding model expects shape
    # (B, C, H, W). Callers that perform batch preprocessing will
    # stack tensors instead of using this helper's unsqueeze.
    tensor = tensor.unsqueeze(0)

    # Move to the chosen device (cpu / cuda) before inference
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
    # Iterate deterministically over sorted filenames so the output order
    # is reproducible (important for embedding → index mapping).
    for name in sorted(os.listdir(image_dir)):
        if not name.lower().endswith(IMAGE_EXTENSIONS):
            continue

        path = os.path.join(image_dir, name)

        try:
            # Load + apply the transform (note: here we append tensors
            # without an extra batch dim — `torch.stack` will create
            # the batch dimension afterwards).
            img = load_image_from_path(path)
            tensor = _preprocess_transform(img)
            tensors.append(tensor)
            image_names.append(name)

        except Exception as e:
            # Don't fail the entire run for a single corrupt image; log
            # and continue. The embedding pipeline handles missing items
            # by checking the final `image_names` list.
            print(f"[WARN] Skipping {name}: {e}")

    if not tensors:
        raise RuntimeError("No valid images found for preprocessing.")

    # Stack along a new batch dimension -> (N, C, H, W) and move to device
    batch = torch.stack(tensors).to(device)
    return batch, image_names