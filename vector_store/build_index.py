"""
build_index.py
--------------
Builds a FAISS vector index from precomputed product embeddings.

Pipeline:
- Load embeddings.npy
- Normalize vectors if cosine similarity is used
- Create FAISS index (cosine or L2)
- Add vectors to index
- Save faiss_index.bin for fast retrieval
"""

import os
import faiss
import numpy as np

# ---------------------------------------------------------
# CONFIG (Matches your folder structure)
# ---------------------------------------------------------

EMBEDDINGS_PATH = "vector_store/embeddings.npy"
INDEX_PATH = "vector_store/faiss_index.bin"

# Similarity metric: "cosine" (recommended) or "l2"
METRIC = "cosine"

# ---------------------------------------------------------
# Load Embeddings
# ---------------------------------------------------------

def load_embeddings(path=EMBEDDINGS_PATH):
    """
    Load precomputed embeddings from disk.
    """
    # Ensure the file exists before attempting to load
    if not os.path.exists(path):
        raise FileNotFoundError(f"Embeddings not found at {path}")

    # Load and cast to float32 which FAISS expects
    embeddings = np.load(path).astype("float32")
    print(f"[INFO] Loaded embeddings: {embeddings.shape}")
    return embeddings


# ---------------------------------------------------------
# Build FAISS Index
# ---------------------------------------------------------

def build_faiss_index(embeddings: np.ndarray, metric: str = "cosine"):
    """
    Create a FAISS index for fast nearest-neighbor search.

    Args:
        embeddings: numpy array of shape (N, D)
        metric: "cosine" or "l2"

    Returns:
        faiss.Index
    """
    dim = embeddings.shape[1]

    if metric == "cosine":
        # Cosine similarity can be computed via inner-product on
        # L2-normalized vectors. Normalize in-place to avoid extra copies.
        faiss.normalize_L2(embeddings)
        # Use inner-product index for normalized vectors
        index = faiss.IndexFlatIP(dim)

    elif metric == "l2":
        # Use standard L2 distance index
        index = faiss.IndexFlatL2(dim)

    else:
        raise ValueError("Metric must be 'cosine' or 'l2'")

    # Add vectors to the index (FAISS keeps an internal copy)
    index.add(embeddings)
    print(f"[INFO] FAISS index built ({metric}), vectors indexed: {index.ntotal}")

    return index


# ---------------------------------------------------------
# Save Index
# ---------------------------------------------------------

def save_index(index, path=INDEX_PATH):
    """
    Save FAISS index to disk.
    """
    # Ensure target directory exists then write the index to disk
    os.makedirs(os.path.dirname(path), exist_ok=True)
    faiss.write_index(index, path)
    print(f"[DONE] FAISS index saved → {path}")


# ---------------------------------------------------------
# Script Entry Point
# ---------------------------------------------------------

if __name__ == "__main__":
    embeddings = load_embeddings()
    index = build_faiss_index(embeddings, metric=METRIC)
    save_index(index)
    # End-to-end: load embeddings, build index, and persist to disk.
