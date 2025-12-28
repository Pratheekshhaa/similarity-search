"""
embed_products.py
-----------------
Extracts visual embeddings for product eyewear images using a pretrained
CNN (ResNet50). These embeddings are later indexed by FAISS for fast
visual similarity search.

Pipeline:
- Load pretrained ImageNet ResNet50
- Preprocess images using `ingestion.preprocess`
- Extract 2048-D embeddings for each image
- Save embeddings + image order mapping for deterministic indexing

Notes:
- The offline embedding extraction is intended to be run once (or when the
    product catalog changes). The output files (`embeddings.npy` and
    `image_names.json`) are required by `vector_store.build_index` and runtime
    search.
"""

import os
import json
import torch
import numpy as np
from tqdm import tqdm
from torchvision import models

from ingestion.preprocess import (
    preprocess_from_path,
    IMAGE_EXTENSIONS
)

# ---------------------------------------------------------
# CONFIG (MATCHES YOUR FOLDER STRUCTURE)
# ---------------------------------------------------------

IMAGE_DIR = "data/product_eyewear/images"
OUTPUT_DIR = "vector_store"

EMBEDDINGS_PATH = os.path.join(OUTPUT_DIR, "embeddings.npy")
IMAGE_NAMES_PATH = os.path.join(OUTPUT_DIR, "image_names.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# Model Loader (ResNet50)
# ---------------------------------------------------------

def load_embedding_model(device="cpu"):
    """
    Loads a pretrained ResNet50 and removes the classification head.

    Output: 2048-D embedding vector. The function returns a model placed on
    the requested `device` and set to evaluation mode. Loading happens once
    during an extraction run to avoid repeated heavy I/O.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    # Replace the final classification layer with an identity to obtain the
    # penultimate feature vector (2048-D for ResNet50).
    model.fc = torch.nn.Identity()
    model = model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------
# Single Image Embedding
# ---------------------------------------------------------

def extract_embedding(model, tensor):
    """
    Runs a preprocessed tensor through the model.

    Args:
        tensor: torch.Tensor of shape (1, 3, 224, 224)

    Returns:
        numpy array of shape (2048,)
    """
    # Inference: disable gradient tracking to save memory and cycles
    with torch.no_grad():
        embedding = model(tensor)

    # Convert to CPU numpy vector and strip batch dimension
    return embedding.squeeze(0).cpu().numpy()


# ---------------------------------------------------------
# Directory Embedding Pipeline
# ---------------------------------------------------------

def embed_directory(image_dir, device="cpu"):
    """
    Extract embeddings for all product images in a directory.

    Returns:
        embeddings: numpy array (N, 2048)
        image_names: list of filenames
    """

    model = load_embedding_model(device=device)

    embeddings = []
    image_names = []

    print(f"[INFO] Extracting embeddings from: {image_dir}")
    print(f"[INFO] Using device: {device}")

    for name in tqdm(sorted(os.listdir(image_dir))):
        if not name.lower().endswith(IMAGE_EXTENSIONS):
            continue

        path = os.path.join(image_dir, name)

        try:
            tensor = preprocess_from_path(path, device=device)
            vector = extract_embedding(model, tensor)

            embeddings.append(vector)
            image_names.append(name)

        except Exception as e:
            print(f"[WARN] Skipping {name}: {e}")

    if not embeddings:
        raise RuntimeError("No embeddings extracted — check dataset path.")

    embeddings = np.vstack(embeddings).astype("float32")

    return embeddings, image_names


# ---------------------------------------------------------
# Save Outputs
# ---------------------------------------------------------

def save_outputs(embeddings, image_names):
    """
    Save embeddings and image order mapping.
    """
    np.save(EMBEDDINGS_PATH, embeddings)

    with open(IMAGE_NAMES_PATH, "w") as f:
        json.dump(image_names, f, indent=2)

    print(f"[DONE] Embeddings saved → {EMBEDDINGS_PATH}")
    print(f"[DONE] Image names saved → {IMAGE_NAMES_PATH}")
    print(f"[DONE] Total images embedded: {len(image_names)}")


# ---------------------------------------------------------
# Script Entry Point
# ---------------------------------------------------------

if __name__ == "__main__":
    embeddings, image_names = embed_directory(
        IMAGE_DIR,
        device=DEVICE
    )
    save_outputs(embeddings, image_names)
