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
    # Load FAISS index once and keep in module-level state so Flask can
    # reuse the same object across requests (avoids repeated disk I/O).
    if _faiss_index is None:
        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError(f"FAISS index not found at {INDEX_PATH}")
        _faiss_index = faiss.read_index(INDEX_PATH)
        print(f"[INFO] FAISS index loaded ({_faiss_index.ntotal} vectors)")

    # Load the image name mapping (index -> filename). The order must
    # match the embeddings used to build the index.
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

    # Ensure the query is a 2D float32 array: (1, D)
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1).astype("float32")

    # If cosine similarity is used during index construction, normalize
    # the query vector so that FAISS's inner-product distances correspond
    # to cosine similarity scores.
    if METRIC == "cosine":
        faiss.normalize_L2(query_embedding)

    # Execute the ANN search. FAISS returns (distances, indices).
    scores, indices = _faiss_index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        # FAISS may return -1 for padded/invalid entries; skip them.
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
