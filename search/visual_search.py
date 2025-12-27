"""
visual_search.py
----------------
Performs nearest-neighbor visual search using a FAISS index.

Pipeline:
- Load FAISS index + image name mapping (once)
- Accept a query embedding (2048-D)
- Normalize if cosine similarity is used
- Return top-K similar product images with scores
"""

import os
import json
import faiss
import numpy as np

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

INDEX_PATH = "vector_store/faiss_index.bin"
IMAGE_NAMES_PATH = "vector_store/image_names.json"
METRIC = "cosine"   # must match build_index.py

# ---------------------------------------------------------
# LOAD ONCE (IMPORTANT FOR FLASK)
# ---------------------------------------------------------

_faiss_index = None
_image_names = None


def _load_resources():
    global _faiss_index, _image_names

    if _faiss_index is None:
        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError(f"FAISS index not found at {INDEX_PATH}")
        _faiss_index = faiss.read_index(INDEX_PATH)
        print(f"[INFO] FAISS index loaded ({_faiss_index.ntotal} vectors)")

    if _image_names is None:
        if not os.path.exists(IMAGE_NAMES_PATH):
            raise FileNotFoundError(f"Image names not found at {IMAGE_NAMES_PATH}")
        with open(IMAGE_NAMES_PATH, "r") as f:
            _image_names = json.load(f)
        print(f"[INFO] Image name mapping loaded ({len(_image_names)})")


# ---------------------------------------------------------
# CORE SEARCH FUNCTION
# ---------------------------------------------------------

def search_similar(query_embedding: np.ndarray, top_k: int = 5):
    """
    Perform nearest-neighbor search.

    Args:
        query_embedding: numpy array (2048,)
        top_k: number of results to return

    Returns:
        List of dicts:
        [
            {"image": "prod_0001.jpg", "score": 0.92},
            ...
        ]
    """

    _load_resources()

    # Ensure correct shape
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1).astype("float32")

    # Normalize for cosine similarity
    if METRIC == "cosine":
        faiss.normalize_L2(query_embedding)

    # FAISS search
    scores, indices = _faiss_index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        results.append({
            "image": _image_names[idx],
            "score": round(float(score), 4)
        })

    return results


# ---------------------------------------------------------
# SCRIPT TEST (OPTIONAL)
# ---------------------------------------------------------

if __name__ == "__main__":
    dummy = np.random.rand(2048).astype("float32")
    results = search_similar(dummy, top_k=5)
    print(results)
